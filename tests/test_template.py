import numpy as np

from orient4d.metrics import angular_error_deg
from orient4d.sim import PHASES, DetectorGeometry, clean_pattern, simulate_pattern
from orient4d.template import build_library, match, preprocess

DET = DetectorGeometry(n_px=48)


def test_library_size_and_labels():
    lib = build_library(DET, step_deg=5.0)
    # hexagonal fold 60 -> 12 templates, square fold 90 -> 18 templates
    assert lib.n_templates == 12 + 18
    assert set(lib.phase_ids.tolist()) == {0, 1}
    assert lib.vectors.shape == (30, 48 * 48)


def test_preprocess_normalises():
    rng = np.random.default_rng(0)
    x = preprocess(rng.poisson(5.0, size=(3, 48, 48)))
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
    assert np.allclose(x.mean(axis=1), 0.0, atol=1e-6)


def test_match_recovers_phase_and_angle_on_clean_patterns():
    lib = build_library(DET, step_deg=1.0)
    for name, theta in [("hexagonal", 33.7), ("square", 71.2)]:
        pat = clean_pattern(PHASES[name], theta, DET) * 1e6
        res = match(pat[None], lib, refine=True)
        assert lib.phase_names[res.phase_id[0]] == name
        err = angular_error_deg(res.theta[0], theta, PHASES[name].fold)
        assert err < 0.15


def test_refinement_beats_grid_snap():
    lib = build_library(DET, step_deg=2.0)
    rng = np.random.default_rng(1)
    thetas = rng.uniform(0, 60, size=10)
    pats = np.array([clean_pattern(PHASES["hexagonal"], t, DET) * 1e6 for t in thetas])
    coarse = match(pats, lib, refine=False)
    fine = match(pats, lib, refine=True)
    err_c = angular_error_deg(coarse.theta, thetas, 60.0).mean()
    err_f = angular_error_deg(fine.theta, thetas, 60.0).mean()
    assert err_f < err_c


def test_match_on_noisy_patterns():
    lib = build_library(DET, step_deg=1.0)
    rng = np.random.default_rng(2)
    pat = simulate_pattern(PHASES["square"], 45.0, DET, dose=500.0, rng=rng)
    res = match(pat[None], lib)
    assert lib.phase_names[res.phase_id[0]] == "square"
    assert angular_error_deg(res.theta[0], 45.0, 90.0) < 2.0
