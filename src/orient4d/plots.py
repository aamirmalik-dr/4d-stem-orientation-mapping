"""Plotting primitives: cyclic orientation colouring, pattern display, maps."""

from __future__ import annotations

import sys

import matplotlib

if "matplotlib.pyplot" not in sys.modules:
    # Headless default for CLI use; a notebook that selected its own backend
    # (e.g. %matplotlib inline) before importing this module keeps it.
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from .sim import FOLDS, PHASE_NAMES, DetectorGeometry, Scan4D
from .virtual import annular_dark_field, bright_field


def orientation_rgb(theta: np.ndarray, phase_id: np.ndarray) -> np.ndarray:
    """Colour an orientation map: hue is the symmetry-reduced angle.

    Hue runs once around the colour wheel over each phase's fold (60 degrees
    for the hexagonal phase, 90 for the square phase). The square phase is
    rendered desaturated so both phase and orientation read from one map.

    Args:
        theta: (rows, cols) orientations in degrees.
        phase_id: (rows, cols) phase index into ``PHASE_NAMES``.

    Returns:
        (rows, cols, 3) float RGB image in [0, 1].
    """
    phase_id = np.asarray(phase_id, dtype=int)
    fold = FOLDS[phase_id]
    hue = (np.asarray(theta) % fold) / fold
    sat = np.where(phase_id == 0, 0.95, 0.45)
    val = np.ones_like(hue)
    return hsv_to_rgb(np.stack([hue, sat, val], axis=-1))


def pattern_extent(det: DetectorGeometry) -> tuple[float, float, float, float]:
    """Imshow extent (left, right, bottom, top) in reciprocal angstrom."""
    return (-det.k_max, det.k_max, -det.k_max, det.k_max)


def show_pattern(ax, pattern: np.ndarray, det: DetectorGeometry, title: str = "") -> None:
    """Display one diffraction pattern on log scale with reciprocal axes."""
    ax.imshow(
        np.log1p(np.asarray(pattern, dtype=float)),
        cmap="inferno",
        origin="lower",
        extent=pattern_extent(det),
    )
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def scene_figure(scan: Scan4D, path: str, n_examples: int = 4, seed: int = 0) -> None:
    """Overview figure: virtual images, ground truth maps, example patterns."""
    rng = np.random.default_rng(seed)
    h, w = scan.data.shape[:2]
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, max(4, n_examples), height_ratios=[1.3, 1])

    vbf = bright_field(scan.data, scan.det)
    vadf = annular_dark_field(scan.data, scan.det)
    for idx, (img, name) in enumerate([(vbf, "virtual BF"), (vadf, "virtual ADF")]):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(orientation_rgb(scan.theta, scan.phase_id))
    ax.set_title("ground-truth orientation", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(scan.phase_id, cmap="coolwarm", vmin=0, vmax=len(PHASE_NAMES) - 1)
    ax.set_title("ground-truth phase", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])

    for e in range(n_examples):
        i, j = int(rng.integers(h)), int(rng.integers(w))
        ax = fig.add_subplot(gs[1, e])
        show_pattern(
            ax,
            scan.data[i, j],
            scan.det,
            title=(
                f"({i},{j}) {PHASE_NAMES[scan.phase_id[i, j]]}\n"
                f"theta={scan.theta[i, j]:.1f} deg, purity={scan.purity[i, j]:.2f}"
            ),
        )
    fig.suptitle(
        f"simulated 4D-STEM scan: {h}x{w} probe positions, "
        f"{scan.params.n_grains} grains, dose {scan.params.dose:g} e/pattern",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def map_figure(
    scan: Scan4D,
    phase_map: np.ndarray,
    theta_map: np.ndarray,
    path: str,
    method_name: str = "",
) -> None:
    """Predicted versus ground-truth orientation and phase maps."""
    from .metrics import angular_error_deg

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    axes[0].imshow(orientation_rgb(theta_map, phase_map))
    axes[0].set_title(f"predicted orientation ({method_name})", fontsize=9)
    axes[1].imshow(orientation_rgb(scan.theta, scan.phase_id))
    axes[1].set_title("ground-truth orientation", fontsize=9)
    axes[2].imshow(phase_map, cmap="coolwarm", vmin=0, vmax=len(PHASE_NAMES) - 1)
    axes[2].set_title("predicted phase", fontsize=9)
    axes[3].imshow(scan.phase_id, cmap="coolwarm", vmin=0, vmax=len(PHASE_NAMES) - 1)
    axes[3].set_title("ground-truth phase", fontsize=9)
    err = angular_error_deg(theta_map, scan.theta, FOLDS[scan.phase_id.astype(int)])
    err = np.where(phase_map == scan.phase_id, err, np.nan)
    im = axes[4].imshow(err, cmap="viridis", vmin=0)
    axes[4].set_title("orientation error (deg)\nwrong phase = white", fontsize=9)
    fig.colorbar(im, ax=axes[4], fraction=0.046)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cluster_figure(scan: Scan4D, labels: np.ndarray, ari: float, path: str) -> None:
    """Unsupervised grain map next to the ground-truth grain map."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    axes[0].imshow(labels, cmap="tab20", interpolation="nearest")
    axes[0].set_title(f"k-means grain clusters (ARI {ari:.3f})", fontsize=10)
    axes[1].imshow(scan.grain_id, cmap="tab20", interpolation="nearest")
    axes[1].set_title("ground-truth grains", fontsize=10)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def virtual_figure(scan: Scan4D, path: str) -> None:
    """Virtual bright-field and annular dark-field images."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, (img, name) in zip(
        axes,
        [
            (bright_field(scan.data, scan.det), "virtual bright field"),
            (annular_dark_field(scan.data, scan.det), "virtual annular dark field"),
        ],
    ):
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
