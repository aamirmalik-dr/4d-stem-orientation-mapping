"""Virtual imaging: synthesize real-space images from 4D-STEM data.

A virtual detector is a mask in the diffraction plane; summing each pattern
over the mask gives one real-space image. Bright field integrates the direct
beam disk, annular dark field integrates a ring of scattered intensity, and a
spot detector integrates a small disk at a chosen reciprocal-space position
(lighting up only the grains that diffract into it).
"""

from __future__ import annotations

import numpy as np

from .sim import DetectorGeometry


def k_radius_grid(det: DetectorGeometry) -> np.ndarray:
    """(n_px, n_px) map of |k| in reciprocal angstrom at each detector pixel."""
    kk = det.pixel_centers()
    ky, kx = np.meshgrid(kk, kk, indexing="ij")
    return np.hypot(ky, kx)


def annular_mask(det: DetectorGeometry, r_in: float, r_out: float) -> np.ndarray:
    """Boolean detector mask for r_in <= |k| < r_out (reciprocal angstrom)."""
    r = k_radius_grid(det)
    return (r >= r_in) & (r < r_out)


def disk_mask(det: DetectorGeometry, center_k: tuple[float, float], radius: float) -> np.ndarray:
    """Boolean detector mask for a disk at ``center_k`` = (ky, kx)."""
    kk = det.pixel_centers()
    ky, kx = np.meshgrid(kk, kk, indexing="ij")
    return np.hypot(ky - center_k[0], kx - center_k[1]) < radius


def virtual_image(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum each pattern over ``mask``; returns a (rows, cols) image."""
    return np.tensordot(data, mask.astype(data.dtype), axes=([2, 3], [0, 1]))


def bright_field(data: np.ndarray, det: DetectorGeometry, radius: float = 0.12) -> np.ndarray:
    """Virtual bright-field image: integrate the direct-beam disk."""
    return virtual_image(data, annular_mask(det, 0.0, radius))


def annular_dark_field(
    data: np.ndarray,
    det: DetectorGeometry,
    r_in: float = 0.30,
    r_out: float = 1.30,
) -> np.ndarray:
    """Virtual annular dark-field image: integrate scattered intensity."""
    return virtual_image(data, annular_mask(det, r_in, r_out))


def spot_dark_field(
    data: np.ndarray,
    det: DetectorGeometry,
    center_k: tuple[float, float],
    radius: float = 0.08,
) -> np.ndarray:
    """Grain-selective dark field from a small disk at one Bragg position."""
    return virtual_image(data, disk_mask(det, center_k, radius))
