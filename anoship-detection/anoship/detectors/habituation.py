"""Habituation-clustering detector (AHSC).

Faithful reference implementation of:

    Hancheng Xiao, WeiFu Zhu, Zhipeng Qiu, Zhixia Zeng, Shi Zhang, Ruliang Xiao.
    "Anti-Drosophila Habituation Clustering for Enhanced Anomaly Detection in
    Data Streams." IEEE ISCIPT 2025. (first author)
    https://ieeexplore.ieee.org/abstract/document/11265546

The AHSC algorithm (assembled in :mod:`anoship.detectors._ahsc`) treats each row
of the window as a point in an evolving data stream and runs the paper's six-step
loop:

1. **Anti-habituation preprocessing** -- a sparse Drosophila olfactory projection
   (``Y = M . X``) followed by a winner-take-all sparsification (mitigating the
   curse of dimensionality) and the anti-habituation enhancement that raises
   intra-cluster cohesion.
2. **k-means++ initialisation** of the micro/macro-cluster model.
3. **Macrocluster-first search** for each point's best microcluster (O(log n)).
4. **Microcluster update** of core/shell counts, center, radius, and weight.
5. **Removal** of abnormal (decayed) microclusters.
6. **Macrocluster (re)construction** from the cross-connected microclusters.

A point that lands outside the nearest core microcluster is anomalous, so the
per-point score is its distance to the best microcluster center relative to that
cluster's radius. The :class:`BaseDetector` machinery then calibrates a threshold
on the anomaly-free baseline and produces the final :class:`AnomalyResult`.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_detector
from ._ahsc.ahsc import AHSC
from .base import BaseDetector

__all__ = ["HabituationClusterDetector"]


@register_detector("habituation")
class HabituationClusterDetector(BaseDetector):
    name = "habituation"

    def __init__(
        self,
        n_clusters: int = 8,
        expansion: int = 40,
        density: float = 0.1,
        wta_frac: float = 0.05,
        alpha: float = 0.1,
        beta: float = 0.05,
        theta: float = 100.0,
        delta: int = 2,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_clusters = int(n_clusters)
        self.expansion = int(expansion)
        self.density = float(density)
        self.wta_frac = float(wta_frac)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.theta = float(theta)
        self.delta = int(delta)
        self.seed = int(seed)
        self._ahsc: AHSC | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._ahsc = AHSC(
            expansion=self.expansion,
            density=self.density,
            wta_frac=self.wta_frac,
            alpha=self.alpha,
            beta=self.beta,
            n_init=self.n_clusters,
            theta=self.theta,
            delta=self.delta,
            seed=self.seed,
        )
        self._ahsc.fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        assert self._ahsc is not None
        return self._ahsc.outlier_scores(X)
