"""A small CNN that maps one diffraction pattern to phase and orientation.

Orientation is regressed as a point on the unit circle of the *symmetry
multiplied* angle: for a pattern with ``n``-fold rotational symmetry the
network predicts ``(cos(n * theta), sin(n * theta))``, which is continuous
across the symmetry wrap (theta = 0 and theta = fold are the same physical
orientation and the same target). One (cos, sin) pair is emitted per phase;
at inference the pair belonging to the predicted phase is decoded.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .sim import PHASE_NAMES, PHASES, DetectorGeometry

NET_INPUT_PX = 64


class OrientPhaseNet(nn.Module):
    """Two-head CNN: phase logits and per-phase (cos, sin) orientation vectors."""

    def __init__(self, n_phases: int = len(PHASE_NAMES), width: int = 24):
        super().__init__()
        self.n_phases = n_phases
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(1, w, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * w, 2 * w, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * w, 4 * w, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head_phase = nn.Linear(4 * w, n_phases)
        self.head_angle = nn.Linear(4 * w, 2 * n_phases)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.features(x).flatten(1)
        return self.head_phase(z), self.head_angle(z)


def net_input(patterns: np.ndarray) -> torch.Tensor:
    """Convert raw count patterns to normalised network input.

    Each pattern is scaled to unit total intensity (dose invariance), square-
    rooted (variance stabilisation), and resampled to 64 x 64 if the detector
    resolution differs (bilinear, which acts as flux-preserving binning up to
    a constant). Returns a (m, 1, 64, 64) float32 tensor.
    """
    x = np.asarray(patterns, dtype=np.float32)
    x = x.reshape(-1, x.shape[-2], x.shape[-1])
    sums = x.sum(axis=(1, 2), keepdims=True)
    x = np.sqrt(x / np.maximum(sums, 1.0))
    t = torch.from_numpy(x).unsqueeze(1)
    if t.shape[-1] != NET_INPUT_PX:
        t = torch.nn.functional.interpolate(
            t, size=(NET_INPUT_PX, NET_INPUT_PX), mode="bilinear", align_corners=False
        )
    return t * 8.0


def angle_targets(theta_deg: np.ndarray, phase_id: np.ndarray) -> np.ndarray:
    """(m, 2) targets (cos, sin) of the symmetry-multiplied angle per sample."""
    orders = np.array([PHASES[n].sym_order for n in PHASE_NAMES])
    phi = np.deg2rad(np.asarray(theta_deg) * orders[np.asarray(phase_id)])
    return np.stack([np.cos(phi), np.sin(phi)], axis=-1).astype(np.float32)


def decode_angle(angle_out: np.ndarray, phase_id: np.ndarray) -> np.ndarray:
    """Decode predicted orientations in degrees from the angle head output.

    Args:
        angle_out: (m, 2 * n_phases) raw head output.
        phase_id: (m,) phase used to select the (cos, sin) pair.

    Returns:
        (m,) orientations in degrees, reduced to each phase's fold.
    """
    phase_id = np.asarray(phase_id, dtype=int)
    rows = np.arange(len(phase_id))
    c = angle_out[rows, 2 * phase_id]
    s = angle_out[rows, 2 * phase_id + 1]
    orders = np.array([PHASES[n].sym_order for n in PHASE_NAMES])
    folds = np.array([PHASES[n].fold for n in PHASE_NAMES])
    theta = np.degrees(np.arctan2(s, c)) % 360.0 / orders[phase_id]
    return theta % folds[phase_id]


@torch.no_grad()
def predict(
    model: OrientPhaseNet,
    patterns: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict (phase_id, theta_deg) for a batch of patterns."""
    model.eval()
    x = net_input(patterns)
    phases, angles = [], []
    for i in range(0, x.shape[0], batch_size):
        logits, ang = model(x[i : i + batch_size])
        phases.append(logits.argmax(dim=1).numpy())
        angles.append(ang.numpy())
    phase_id = np.concatenate(phases)
    theta = decode_angle(np.concatenate(angles), phase_id)
    return phase_id, theta


def map_scan(
    model: OrientPhaseNet, data: np.ndarray, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Predict phase and orientation maps for a (rows, cols, n_px, n_px) scan."""
    h, w = data.shape[:2]
    phase_id, theta = predict(model, data.reshape(h * w, *data.shape[2:]), batch_size)
    return phase_id.reshape(h, w), theta.reshape(h, w)


def save_model(model: OrientPhaseNet, path: str) -> None:
    """Save model weights and width metadata."""
    width = model.head_phase.in_features // 4
    torch.save({"state_dict": model.state_dict(), "width": width}, path)


def load_model(path: str) -> OrientPhaseNet:
    """Load a model saved with :func:`save_model` (weights_only for safety)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = OrientPhaseNet(width=int(payload["width"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def default_detector() -> DetectorGeometry:
    """The detector geometry the committed model was trained on."""
    return DetectorGeometry()
