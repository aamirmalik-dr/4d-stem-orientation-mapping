"""Template matching against an orientation library.

The classical baseline. A library of noise-free patterns is simulated on a
regular orientation grid for each phase; each measured pattern is scored by
normalised cross-correlation (on square-root intensities, which stabilises
Poisson noise and tames the direct beam) against every template, and the best
match assigns phase and orientation. A three-point parabolic fit over the
correlation of the neighbouring library angles refines the orientation below
the library step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .sim import PHASES, DetectorGeometry, clean_pattern


def preprocess(patterns: np.ndarray) -> np.ndarray:
    """Flatten patterns to zero-mean, unit-norm sqrt-intensity vectors.

    Args:
        patterns: (..., n_px, n_px) counts or intensities, non-negative.

    Returns:
        (m, n_px * n_px) float32 array of normalised feature vectors.
    """
    x = np.sqrt(np.maximum(np.asarray(patterns, dtype=np.float64), 0.0))
    x = x.reshape(-1, x.shape[-2] * x.shape[-1])
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


@dataclass
class TemplateLibrary:
    """A stack of preprocessed templates with their phase and orientation labels.

    Attributes:
        vectors: (n_templates, n_px * n_px) preprocessed template vectors.
        phase_ids: (n_templates,) index into the library's phase name order.
        thetas: (n_templates,) template orientation in degrees.
        phase_names: Phase name per phase id.
        step_deg: Library angular step in degrees.
        det: Detector geometry the templates were rendered on.
    """

    vectors: np.ndarray
    phase_ids: np.ndarray
    thetas: np.ndarray
    phase_names: tuple[str, ...]
    step_deg: float
    det: DetectorGeometry

    @property
    def n_templates(self) -> int:
        return int(self.vectors.shape[0])


def build_library(
    det: DetectorGeometry,
    step_deg: float = 0.5,
    phase_names: tuple[str, ...] = ("hexagonal", "square"),
) -> TemplateLibrary:
    """Simulate a noise-free orientation library on a regular angular grid."""
    vecs, pids, thetas = [], [], []
    for pid, name in enumerate(phase_names):
        phase = PHASES[name]
        for theta in np.arange(0.0, phase.fold, step_deg):
            vecs.append(clean_pattern(phase, float(theta), det))
            pids.append(pid)
            thetas.append(float(theta))
    return TemplateLibrary(
        vectors=preprocess(np.array(vecs)),
        phase_ids=np.array(pids, dtype=np.int64),
        thetas=np.array(thetas, dtype=np.float64),
        phase_names=phase_names,
        step_deg=float(step_deg),
        det=det,
    )


@dataclass
class MatchResult:
    """Template-matching output for a batch of patterns.

    Attributes:
        phase_id: (m,) best-match phase index.
        theta: (m,) refined orientation in degrees within the phase fold.
        score: (m,) best normalised cross-correlation.
        seconds_per_pattern: Mean wall-clock matching cost per pattern.
    """

    phase_id: np.ndarray
    theta: np.ndarray
    score: np.ndarray
    seconds_per_pattern: float


def match(patterns: np.ndarray, library: TemplateLibrary, refine: bool = True) -> MatchResult:
    """Match patterns against the library; optionally refine sub-step.

    Args:
        patterns: (..., n_px, n_px) measured patterns.
        library: Orientation library from :func:`build_library`.
        refine: If True, parabolic interpolation over the two angular
            neighbours of the best template sharpens the orientation estimate.

    Returns:
        A :class:`MatchResult` with per-pattern phase, orientation and score.
    """
    t0 = time.perf_counter()
    x = preprocess(patterns)
    scores = x @ library.vectors.T
    best = np.argmax(scores, axis=1)
    phase_id = library.phase_ids[best]
    theta = library.thetas[best].copy()
    best_score = scores[np.arange(len(best)), best]

    if refine:
        for pid in np.unique(phase_id):
            block = np.where(library.phase_ids == pid)[0]
            order = np.argsort(library.thetas[block])
            block = block[order]
            n_block = len(block)
            pos_of = {int(t_idx): p for p, t_idx in enumerate(block)}
            fold = PHASES[library.phase_names[pid]].fold
            rows = np.where(phase_id == pid)[0]
            for r in rows:
                p = pos_of[int(best[r])]
                s_c = scores[r, block[p]]
                s_l = scores[r, block[(p - 1) % n_block]]
                s_r = scores[r, block[(p + 1) % n_block]]
                denom = s_l - 2.0 * s_c + s_r
                if denom < -1e-12:
                    delta = 0.5 * (s_l - s_r) / denom
                    delta = float(np.clip(delta, -0.5, 0.5))
                    theta[r] = (library.thetas[block[p]] + delta * library.step_deg) % fold

    elapsed = time.perf_counter() - t0
    return MatchResult(
        phase_id=phase_id,
        theta=theta,
        score=best_score,
        seconds_per_pattern=elapsed / max(1, x.shape[0]),
    )


def map_scan(
    data: np.ndarray, library: TemplateLibrary, refine: bool = True
) -> tuple[np.ndarray, np.ndarray, MatchResult]:
    """Match a (rows, cols, n_px, n_px) scan; returns phase and theta maps."""
    h, w = data.shape[:2]
    res = match(data.reshape(h * w, *data.shape[2:]), library, refine=refine)
    return res.phase_id.reshape(h, w), res.theta.reshape(h, w), res
