"""Scoring: symmetry-aware orientation error, phase accuracy, grain agreement."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .sim import FOLDS


def angular_error_deg(
    pred_theta: np.ndarray,
    true_theta: np.ndarray,
    fold: np.ndarray | float,
) -> np.ndarray:
    """Symmetry-aware absolute angular error in degrees.

    Both angles are compared modulo ``fold`` (the symmetry-reduced range,
    e.g. 60 for a 6-fold pattern), so 0.5 and 59.5 degrees differ by 1.0.

    Args:
        pred_theta: Predicted orientations in degrees.
        true_theta: Ground-truth orientations in degrees.
        fold: Symmetry-reduced range per element (scalar or broadcastable).

    Returns:
        Elementwise absolute error in degrees, in [0, fold / 2].
    """
    d = np.mod(np.asarray(pred_theta) - np.asarray(true_theta), fold)
    return np.minimum(d, fold - d)


def orientation_phase_metrics(
    pred_theta: np.ndarray,
    pred_phase: np.ndarray,
    true_theta: np.ndarray,
    true_phase: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Score an orientation + phase map against ground truth.

    Orientation error is computed only where the predicted phase matches the
    true phase (a wrong-phase pixel has no meaningful angle comparison); the
    phase error rate is reported alongside. ``mask`` restricts scoring, e.g.
    to grain-interior (high-purity) positions.

    Returns:
        Dict with phase_accuracy, n_scored, and orientation error statistics
        (mean, median, p90, rms, frac_within_1deg) over phase-correct pixels.
    """
    pred_theta = np.asarray(pred_theta).ravel()
    pred_phase = np.asarray(pred_phase).ravel()
    true_theta = np.asarray(true_theta).ravel()
    true_phase = np.asarray(true_phase).ravel()
    sel = np.ones_like(true_phase, dtype=bool) if mask is None else np.asarray(mask).ravel()

    phase_ok = pred_phase == true_phase
    phase_acc = float(np.mean(phase_ok[sel])) if sel.any() else float("nan")
    scored = sel & phase_ok
    out = {
        "phase_accuracy": phase_acc,
        "n_selected": int(sel.sum()),
        "n_scored": int(scored.sum()),
    }
    if scored.any():
        err = angular_error_deg(
            pred_theta[scored], true_theta[scored], FOLDS[true_phase[scored].astype(int)]
        )
        out.update(
            {
                "orientation_mae_deg": float(np.mean(err)),
                "orientation_median_deg": float(np.median(err)),
                "orientation_p90_deg": float(np.percentile(err, 90)),
                "orientation_rms_deg": float(np.sqrt(np.mean(err**2))),
                "frac_within_1deg": float(np.mean(err <= 1.0)),
            }
        )
    else:
        for key in (
            "orientation_mae_deg",
            "orientation_median_deg",
            "orientation_p90_deg",
            "orientation_rms_deg",
            "frac_within_1deg",
        ):
            out[key] = float("nan")
    return out


def grain_agreement(pred_labels: np.ndarray, true_grain_id: np.ndarray) -> dict:
    """Clustering agreement between predicted grain labels and ground truth."""
    pred = np.asarray(pred_labels).ravel()
    true = np.asarray(true_grain_id).ravel()
    return {
        "ari": float(adjusted_rand_score(true, pred)),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "n_pred_clusters": int(len(np.unique(pred))),
        "n_true_grains": int(len(np.unique(true))),
    }
