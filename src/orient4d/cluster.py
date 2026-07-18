"""Unsupervised grain segmentation: PCA features plus k-means over patterns.

Positions in the same grain share a phase and orientation, so their patterns
are nearly identical up to noise; clustering the patterns recovers the grain
map without any labels or forward model. Two grains whose orientations happen
to fall within the pattern noise floor of each other are physically
indistinguishable and will merge; the adjusted Rand index against ground
truth reports exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def pattern_features(
    data: np.ndarray,
    n_components: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """PCA scores of sqrt-normalised patterns.

    Args:
        data: (rows, cols, n_px, n_px) scan or (m, n_px, n_px) pattern stack.
        n_components: Number of principal components to keep.
        seed: Random state for the (randomised) PCA solver.

    Returns:
        (m, n_components) float64 feature matrix.
    """
    x = np.asarray(data, dtype=np.float64)
    x = x.reshape(-1, x.shape[-2] * x.shape[-1])
    x = np.sqrt(np.maximum(x, 0.0))
    sums = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(sums, 1e-12)
    n_components = min(n_components, x.shape[0] - 1, x.shape[1])
    return PCA(n_components=n_components, random_state=seed).fit_transform(x)


@dataclass
class ClusterResult:
    """Clustering output.

    Attributes:
        labels: (rows, cols) or (m,) cluster label per position.
        k: Number of clusters used.
        k_scores: Silhouette score per candidate k (empty if k was fixed).
        inertia: Final k-means inertia.
    """

    labels: np.ndarray
    k: int
    k_scores: dict[int, float]
    inertia: float


def cluster_grains(
    data: np.ndarray,
    k: int | None = None,
    k_range: tuple[int, int] = (2, 16),
    n_components: int = 20,
    seed: int = 0,
) -> ClusterResult:
    """Cluster scan positions into grains.

    Args:
        data: (rows, cols, n_px, n_px) scan.
        k: Number of clusters; if None, chosen by silhouette over ``k_range``.
        k_range: Inclusive candidate range for automatic k selection.
        n_components: PCA dimensionality of the pattern features.
        seed: Random state for PCA and k-means.

    Returns:
        A :class:`ClusterResult`; labels are reshaped to the scan grid when
        the input is 4D.
    """
    feats = pattern_features(data, n_components=n_components, seed=seed)
    k_scores: dict[int, float] = {}
    if k is None:
        for kk in range(k_range[0], k_range[1] + 1):
            km = KMeans(n_clusters=kk, n_init=4, random_state=seed).fit(feats)
            k_scores[kk] = float(silhouette_score(feats, km.labels_))
        k = max(k_scores, key=lambda kk: k_scores[kk])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(feats)
    labels = km.labels_
    if np.asarray(data).ndim == 4:
        labels = labels.reshape(np.asarray(data).shape[:2])
    return ClusterResult(labels=labels, k=int(k), k_scores=k_scores, inertia=float(km.inertia_))
