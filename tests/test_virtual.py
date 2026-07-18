import numpy as np

from orient4d.sim import PHASES, DetectorGeometry, clean_pattern, lattice_reflections
from orient4d.virtual import (
    annular_dark_field,
    annular_mask,
    bright_field,
    disk_mask,
    spot_dark_field,
    virtual_image,
)

DET = DetectorGeometry(n_px=48)


def test_masks_partition_sensibly():
    bf = annular_mask(DET, 0.0, 0.12)
    adf = annular_mask(DET, 0.30, 1.30)
    assert bf.sum() > 0 and adf.sum() > 0
    assert not np.any(bf & adf)


def test_virtual_image_sums_counts():
    data = np.ones((2, 3, 48, 48), dtype=np.float32)
    mask = disk_mask(DET, (0.0, 0.0), 0.2)
    img = virtual_image(data, mask)
    assert img.shape == (2, 3)
    assert np.allclose(img, mask.sum())


def test_bright_field_sees_direct_beam():
    pat = clean_pattern(PHASES["square"], 0.0, DET)[None, None]
    bf = bright_field(pat, DET, radius=0.12)[0, 0]
    adf = annular_dark_field(pat, DET)[0, 0]
    # Direct beam carries direct_fraction of the mass in a small disk.
    assert bf > 0.8 * DET.direct_fraction
    assert adf > 0.0


def test_spot_dark_field_is_orientation_selective():
    phase = PHASES["hexagonal"]
    g, _ = lattice_reflections(phase, DET.k_max)
    g0 = g[np.argmin(np.linalg.norm(g, axis=1))]
    # Aperture on a first-ring spot of the unrotated pattern; (ky, kx) order.
    center = (float(g0[1]), float(g0[0]))
    matched = clean_pattern(phase, 0.0, DET)[None, None]
    rotated = clean_pattern(phase, 30.0, DET)[None, None]
    v_match = spot_dark_field(matched, DET, center, radius=0.06)[0, 0]
    v_rot = spot_dark_field(rotated, DET, center, radius=0.06)[0, 0]
    assert v_match > 3.0 * v_rot
