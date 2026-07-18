import numpy as np
import pytest

from orient4d.io import load_external, load_scan, save_scan
from orient4d.sim import DetectorGeometry, SceneParams, simulate_scene_scan


def _small_scan():
    params = SceneParams(scan_shape=(6, 6), n_grains=2, dose=100.0)
    return simulate_scene_scan(params, DetectorGeometry(n_px=24), seed=3)


def test_save_load_roundtrip(tmp_path):
    scan = _small_scan()
    path = tmp_path / "scan.npz"
    save_scan(scan, path)
    loaded = load_scan(path)
    assert np.allclose(loaded.data, scan.data)
    assert np.array_equal(loaded.grain_id, scan.grain_id)
    assert np.array_equal(loaded.phase_id, scan.phase_id)
    assert np.allclose(loaded.theta, scan.theta, atol=1e-6)
    assert loaded.det == scan.det
    assert loaded.params == scan.params


def test_load_external_npy_and_npz(tmp_path):
    arr = np.random.default_rng(0).poisson(2.0, size=(4, 5, 16, 16)).astype(np.float32)
    p_npy = tmp_path / "d.npy"
    np.save(p_npy, arr)
    data, det = load_external(p_npy, k_max=2.0)
    assert data.shape == arr.shape
    assert det.n_px == 16
    assert det.k_max == 2.0
    p_npz = tmp_path / "d.npz"
    np.savez(p_npz, data=arr)
    data2, _ = load_external(p_npz)
    assert np.allclose(data2, arr)


def test_load_external_crops_non_square(tmp_path):
    arr = np.zeros((3, 3, 16, 20), dtype=np.float32)
    p = tmp_path / "d.npy"
    np.save(p, arr)
    data, det = load_external(p)
    assert data.shape == (3, 3, 16, 16)
    assert det.n_px == 16


def test_load_external_rejects_bad_shapes(tmp_path):
    p = tmp_path / "bad.npy"
    np.save(p, np.zeros((4, 4)))
    with pytest.raises(ValueError):
        load_external(p)
    p2 = tmp_path / "bad.npz"
    np.savez(p2, other=np.zeros((2, 2, 4, 4)))
    with pytest.raises(ValueError):
        load_external(p2)
