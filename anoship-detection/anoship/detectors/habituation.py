"""Habituation-clustering detector.

Reference implementation of the core idea from:

    Hancheng Xiao, WeiFu Zhu, Zhipeng Qiu, Zhixia Zeng, Shi Zhang, Ruliang Xiao.
    "Anti-Drosophila Habituation Clustering for Enhanced Anomaly Detection in
    Data Streams." IEEE ISCIPT 2025. (first author)
    https://ieeexplore.ieee.org/abstract/document/11265546

Core idea
---------
Borrowing the biological notion of *habituation* (a Drosophila repeatedly
exposed to a harmless stimulus stops responding to it), the detector learns
clusters of recurring normal patterns and *suppresses* the anomaly response for
patterns it has grown familiar with. Rare / novel patterns -- those far from any
habituated cluster -- retain a high response. This sharply reduces false alarms
on repetitive but noisy streams, the dominant failure mode of naive distance
detectors in continuous monitoring.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_detector
from .base import BaseDetector

__all__ = ["HabituationClusterDetector"]


@register_detector("habituation")
class HabituationClusterDetector(BaseDetector):
    name = "habituation"

    def __init__(
        self,
        n_clusters: int = 8,
        n_iter: int = 10,
        habituation: float = 0.7,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_clusters = int(n_clusters)
        self.n_iter = int(n_iter)
        self.habituation = float(habituation)
        self.seed = int(seed)
        self._centroids: np.ndarray | None = None
        self._familiarity: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        k = min(self.n_clusters, len(X))
        idx = rng.choice(len(X), size=k, replace=False)
        centroids = X[idx].copy()

        # Lloyd's iterations (k-means-lite).
        for _ in range(self.n_iter):
            assign = self._assign(X, centroids)
            for c in range(k):
                members = X[assign == c]
                if len(members):
                    centroids[c] = members.mean(axis=0)
        assign = self._assign(X, centroids)

        # Familiarity == habituation strength: how often each cluster fires on
        # the normal baseline, normalized to [0, 1].
        counts = np.bincount(assign, minlength=k).astype(float)
        familiarity = counts / counts.max() if counts.max() > 0 else counts
        self._centroids = centroids
        self._familiarity = familiarity

    @staticmethod
    def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # (T, k) distance matrix -> nearest centroid index per row.
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        return d.argmin(axis=1)

    def _score(self, X: np.ndarray) -> np.ndarray:
        assert self._centroids is not None and self._familiarity is not None
        d = np.linalg.norm(X[:, None, :] - self._centroids[None, :, :], axis=2)
        nearest = d.argmin(axis=1)
        min_dist = d[np.arange(len(X)), nearest]
        # Habituated (familiar) clusters dampen the response; novel ones do not.
        suppression = 1.0 - self.habituation * self._familiarity[nearest]
        return min_dist * suppression
