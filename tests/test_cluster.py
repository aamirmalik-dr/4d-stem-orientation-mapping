import numpy as np

from orient4d.cluster import cluster_grains, pattern_features
from orient4d.metrics import grain_agreement
from orient4d.sim import PHASES, DetectorGeometry, simulate_pattern


def _two_grain_stack(n_per: int = 40, dose: float = 2000.0):
    det = DetectorGeometry(n_px=32)
    rng = np.random.default_rng(0)
    a = [simulate_pattern(PHASES["hexagonal"], 10.0, det, dose, rng) for _ in range(n_per)]
    b = [simulate_pattern(PHASES["square"], 40.0, det, dose, rng) for _ in range(n_per)]
    data = np.array(a + b).reshape(2 * n_per, 32, 32)
    labels = np.repeat([0, 1], n_per)
    return data, labels


def test_pattern_features_shape():
    data, _ = _two_grain_stack(10)
    feats = pattern_features(data, n_components=5)
    assert feats.shape == (20, 5)


def test_clustering_separates_two_obvious_grains():
    data, true = _two_grain_stack()
    res = cluster_grains(data, k=2, seed=0)
    assert grain_agreement(res.labels, true)["ari"] == 1.0


def test_auto_k_selects_two():
    data, _ = _two_grain_stack()
    res = cluster_grains(data, k=None, k_range=(2, 5), seed=0)
    assert res.k == 2
    assert set(res.k_scores.keys()) == {2, 3, 4, 5}
