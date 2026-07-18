import numpy as np
import pytest

from orient4d.sim import (
    PHASES,
    DetectorGeometry,
    SceneParams,
    clean_pattern,
    grain_label_map,
    lattice_reflections,
    make_scene,
    simulate_pattern,
    simulate_scene_scan,
)

DET = DetectorGeometry(n_px=48)


def test_clean_pattern_is_normalised():
    for name in PHASES:
        p = clean_pattern(PHASES[name], 12.3, DET)
        assert p.shape == (48, 48)
        assert np.all(p >= 0)
        assert p.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", ["hexagonal", "square"])
def test_rotational_symmetry(name):
    phase = PHASES[name]
    a = clean_pattern(phase, 17.0, DET)
    b = clean_pattern(phase, 17.0 + phase.fold, DET)
    assert np.allclose(a, b, atol=1e-10)


def test_distinct_orientations_give_distinct_patterns():
    phase = PHASES["hexagonal"]
    a = clean_pattern(phase, 5.0, DET)
    b = clean_pattern(phase, 25.0, DET)
    assert np.abs(a - b).max() > 1e-4


def test_first_ring_radii_match_lattice():
    g_hex, _ = lattice_reflections(PHASES["hexagonal"], 1.35)
    r_hex = np.linalg.norm(g_hex, axis=1).min()
    # d_10 of a hexagonal net with a = 2.46 is a * sqrt(3)/2 = 2.130 angstrom
    assert r_hex == pytest.approx(1.0 / 2.130, rel=1e-3)
    g_sq, _ = lattice_reflections(PHASES["square"], 1.35)
    r_sq = np.linalg.norm(g_sq, axis=1).min()
    assert r_sq == pytest.approx(1.0 / 2.03, rel=1e-3)


def test_honeycomb_structure_factor_modulation():
    # Graphene basis: |F|^2 is 4x the single-atom value when h + 2k = 0 mod 3.
    g, inten = lattice_reflections(PHASES["hexagonal"], 1.35)
    r = np.linalg.norm(g, axis=1)
    first_ring = inten[np.isclose(r, r.min(), rtol=1e-3)]
    second_ring_r = np.unique(np.round(r, 5))[1]
    second_ring = inten[np.isclose(r, second_ring_r, rtol=1e-3)]
    assert len(first_ring) == 6
    assert len(second_ring) == 6
    # First ring (h+2k not divisible by 3) is the weak family here.
    ratio = second_ring.mean() / first_ring.mean()
    envelope = np.exp(-2 * 0.7 * (second_ring_r**2 - r.min() ** 2))
    assert ratio == pytest.approx(4.0 * envelope, rel=0.05)


def test_poisson_dose_budget():
    rng = np.random.default_rng(0)
    pats = [simulate_pattern(PHASES["square"], 10.0, DET, 500.0, rng) for _ in range(20)]
    mean_total = np.mean([p.sum() for p in pats])
    assert mean_total == pytest.approx(500.0, rel=0.05)


def test_make_scene_properties():
    params = SceneParams(scan_shape=(24, 24), n_grains=5)
    scene = make_scene(params, np.random.default_rng(1))
    assert len(scene.grains) == 5
    for g in scene.grains:
        assert 0.0 <= g.theta < PHASES[g.phase_name].fold


def test_simulate_scan_shapes_and_truth():
    params = SceneParams(scan_shape=(8, 8), n_grains=3, dose=100.0)
    scan = simulate_scene_scan(params, DetectorGeometry(n_px=32), seed=7)
    assert scan.data.shape == (8, 8, 32, 32)
    assert scan.grain_id.shape == (8, 8)
    assert np.all(scan.purity > 0) and np.all(scan.purity <= 1.0)
    folds = np.array([PHASES[n].fold for n in scan.params.phase_names])
    assert np.all(scan.theta >= 0)
    assert np.all(scan.theta < folds[scan.phase_id] + 1e-6)
    # Dominant grain agrees with the Voronoi cell at the probe position.
    yy, xx = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    pts = np.stack([yy, xx], axis=-1).astype(float)
    labels = grain_label_map(scan.scene, pts.reshape(-1, 2)).reshape(8, 8)
    pure = scan.purity > 0.99
    assert np.all(scan.grain_id[pure] == labels[pure])


def test_scale_and_center_move_the_pattern():
    phase = PHASES["square"]
    base = clean_pattern(phase, 0.0, DET)
    scaled = clean_pattern(phase, 0.0, DET, scale=1.05)
    shifted = clean_pattern(phase, 0.0, DET, center_px=(2.0, 0.0))
    assert np.abs(base - scaled).max() > 1e-4
    assert np.abs(base - shifted).max() > 1e-4
