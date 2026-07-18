"""Training for the orientation + phase CNN on freshly simulated patterns.

Every training sample is simulated on the fly with full domain randomisation:
random phase, uniform orientation, log-uniform dose, camera-length and descan
jitter, randomised direct-beam and background budgets, and (with some
probability) a two-grain mixture with a dominant weight, mimicking probe
positions on a grain boundary. The supervision target of a mixture is its
dominant grain, matching how the benchmark scores boundary pixels.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
from torch import nn

from .net import OrientPhaseNet, angle_targets, decode_angle, net_input
from .sim import PHASE_NAMES, PHASES, DetectorGeometry, clean_pattern


@dataclass(frozen=True)
class TrainSettings:
    """Hyperparameters and randomisation ranges for training.

    Attributes:
        steps: Number of optimiser steps.
        batch_size: Patterns per step.
        lr: Adam learning rate.
        dose_range: (low, high) of the log-uniform dose draw.
        mixture_prob: Probability a sample is a two-grain boundary mixture.
        mixture_dominant_range: Uniform range of the dominant-grain weight.
        scale_jitter: Camera-length jitter sigma (fractional).
        center_jitter: Descan jitter sigma in detector pixels.
        direct_range: Uniform range of the direct-beam dose fraction.
        background_range: Uniform range of the background dose fraction.
        angle_loss_weight: Weight of the orientation MSE term against the CE.
        val_every: Validation cadence in steps.
        val_size: Number of fixed validation patterns.
        seed: Seed for the training stream and validation set.
    """

    steps: int = 1500
    batch_size: int = 64
    lr: float = 1e-3
    dose_range: tuple[float, float] = (10.0, 3000.0)
    mixture_prob: float = 0.3
    mixture_dominant_range: tuple[float, float] = (0.55, 0.95)
    scale_jitter: float = 0.02
    center_jitter: float = 0.4
    direct_range: tuple[float, float] = (0.15, 0.35)
    background_range: tuple[float, float] = (0.08, 0.25)
    angle_loss_weight: float = 2.0
    val_every: int = 100
    val_size: int = 512
    seed: int = 20260718


def sample_batch(
    settings: TrainSettings,
    det: DetectorGeometry,
    rng: np.random.Generator,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate ``n`` randomised (pattern, phase_id, theta) training samples."""
    patterns = np.zeros((n, det.n_px, det.n_px), dtype=np.float32)
    phase_ids = np.zeros(n, dtype=np.int64)
    thetas = np.zeros(n, dtype=np.float64)
    lo, hi = settings.dose_range
    for i in range(n):
        pid = int(rng.integers(len(PHASE_NAMES)))
        phase = PHASES[PHASE_NAMES[pid]]
        theta = float(rng.uniform(0.0, phase.fold))
        dose = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        det_i = replace(
            det,
            direct_fraction=float(rng.uniform(*settings.direct_range)),
            background_fraction=float(rng.uniform(*settings.background_range)),
        )
        scale = float(1.0 + rng.normal(0.0, settings.scale_jitter))
        center = tuple(rng.normal(0.0, settings.center_jitter, size=2))
        expected = clean_pattern(phase, theta, det_i, scale=scale, center_px=center)
        if rng.uniform() < settings.mixture_prob:
            w = float(rng.uniform(*settings.mixture_dominant_range))
            pid2 = int(rng.integers(len(PHASE_NAMES)))
            phase2 = PHASES[PHASE_NAMES[pid2]]
            theta2 = float(rng.uniform(0.0, phase2.fold))
            other = clean_pattern(phase2, theta2, det_i, scale=scale, center_px=center)
            expected = w * expected + (1.0 - w) * other
        patterns[i] = rng.poisson(expected * dose).astype(np.float32)
        phase_ids[i] = pid
        thetas[i] = theta
    return patterns, phase_ids, thetas


def train(
    settings: TrainSettings | None = None,
    det: DetectorGeometry | None = None,
    width: int = 24,
    progress: bool = True,
) -> tuple[OrientPhaseNet, dict]:
    """Train an :class:`OrientPhaseNet` from scratch; returns (model, history).

    History contains per-validation-step loss, validation phase accuracy and
    validation mean angular error, plus the settings and wall-clock time.
    """
    settings = settings or TrainSettings()
    det = det or DetectorGeometry()
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    val_rng = np.random.default_rng(settings.seed + 1)
    val_pat, val_phase, val_theta = sample_batch(settings, det, val_rng, settings.val_size)

    model = OrientPhaseNet(width=width)
    opt = torch.optim.Adam(model.parameters(), lr=settings.lr)
    ce = nn.CrossEntropyLoss()
    history: dict = {"settings": asdict(settings), "steps": [], "width": width}
    t0 = time.perf_counter()

    for step in range(1, settings.steps + 1):
        model.train()
        pats, pids, thetas = sample_batch(settings, det, rng, settings.batch_size)
        x = net_input(pats)
        logits, angles = model(x)
        target_phase = torch.from_numpy(pids)
        target_angle = torch.from_numpy(angle_targets(thetas, pids))
        rows = torch.arange(len(pids))
        pred_pair = torch.stack(
            [angles[rows, 2 * target_phase], angles[rows, 2 * target_phase + 1]], dim=1
        )
        loss = ce(logits, target_phase) + settings.angle_loss_weight * torch.mean(
            (pred_pair - target_angle) ** 2
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % settings.val_every == 0 or step == settings.steps:
            va = evaluate(model, val_pat, val_phase, val_theta)
            entry = {"step": step, "train_loss": float(loss.item()), **va}
            history["steps"].append(entry)
            if progress:
                print(
                    f"step {step:5d}  loss {entry['train_loss']:.4f}  "
                    f"val phase acc {va['val_phase_accuracy']:.3f}  "
                    f"val angle mae {va['val_angle_mae_deg']:.3f} deg"
                )

    history["train_seconds"] = time.perf_counter() - t0
    return model, history


@torch.no_grad()
def evaluate(
    model: OrientPhaseNet,
    patterns: np.ndarray,
    phase_ids: np.ndarray,
    thetas: np.ndarray,
) -> dict:
    """Validation phase accuracy and angular error (scored at the true phase)."""
    from .metrics import angular_error_deg
    from .sim import FOLDS

    model.eval()
    x = net_input(patterns)
    logits, angles = model(x)
    pred_phase = logits.argmax(dim=1).numpy()
    theta_at_true = decode_angle(angles.numpy(), phase_ids)
    err = angular_error_deg(theta_at_true, thetas, FOLDS[phase_ids.astype(int)])
    return {
        "val_phase_accuracy": float(np.mean(pred_phase == phase_ids)),
        "val_angle_mae_deg": float(np.mean(err)),
        "val_angle_median_deg": float(np.median(err)),
    }


def save_history(history: dict, path: str) -> None:
    """Write the training history dict as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
