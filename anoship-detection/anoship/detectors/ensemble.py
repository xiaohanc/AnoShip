"""Ensemble detector: fuse several detectors behind one interface."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.errors import NotFittedError
from ..core.interfaces import Detector, ThresholdStrategy
from ..core.types import AnomalyResult
from ..scoring.fusion import fuse_scores
from ..scoring.thresholds import SigmaThreshold

__all__ = ["EnsembleDetector"]


class EnsembleDetector(Detector):
    """Combines member detectors by fusing their (rescaled) score streams.

    Ensembling trades a little latency for robustness: a regression that one
    detector family misses is often caught by another, which is exactly why
    production health gates rarely rely on a single signal.
    """

    name = "ensemble"

    def __init__(
        self,
        detectors: Sequence[Detector],
        method: str = "mean",
        weights: Optional[Sequence[float]] = None,
        threshold: Optional[ThresholdStrategy] = None,
        min_anomaly_rate: float = 0.05,
    ) -> None:
        if not detectors:
            raise ValueError("ensemble requires at least one detector")
        self.detectors: List[Detector] = list(detectors)
        self.method = method
        self.weights = weights
        self.threshold_strategy = threshold or SigmaThreshold()
        self.min_anomaly_rate = float(min_anomaly_rate)
        self._fitted = False

    def fit(self, X: np.ndarray) -> "EnsembleDetector":
        for det in self.detectors:
            det.fit(X)
        base = self._fused(X)
        self.threshold_strategy.fit(base)
        self._fitted = True
        return self

    def _fused(self, window: np.ndarray) -> np.ndarray:
        streams = [det.score(window) for det in self.detectors]
        return fuse_scores(streams, method=self.method, weights=self.weights)

    def score(self, window: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("ensemble is not fitted")
        return self._fused(window)

    def is_anomaly(self, window: np.ndarray) -> AnomalyResult:
        scores = self.score(window)
        thr = float(self.threshold_strategy.threshold(scores))
        mask = scores > thr
        rate = float(mask.mean()) if scores.size else 0.0
        aggregate = float(scores.max()) if scores.size else 0.0
        attribution = self._merge_attribution(window, mask)
        return AnomalyResult(
            scores=scores,
            score=aggregate,
            threshold=thr,
            is_anomaly=rate >= self.min_anomaly_rate,
            anomaly_rate=rate,
            attribution=attribution,
            detector=self.name,
            meta={"members": [d.name for d in self.detectors]},
        )

    def _merge_attribution(
        self, window: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        n = 0
        for det in self.detectors:
            res = det.is_anomaly(window)
            if res.attribution:
                n += 1
                for k, v in res.attribution.items():
                    merged[k] = merged.get(k, 0.0) + v
        if n == 0:
            return {}
        total = sum(merged.values())
        if total <= 0:
            return merged
        return {k: v / total for k, v in merged.items()}
