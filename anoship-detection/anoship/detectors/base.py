"""Base detector: shared fit / score / threshold / attribution machinery.

Concrete detectors implement two small hooks -- ``_fit`` and ``_score`` -- and
inherit a consistent thresholding and root-cause-attribution pipeline. This keeps
each published method's file focused on its actual idea.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.errors import NotFittedError
from ..core.interfaces import Detector, ThresholdStrategy
from ..core.types import AnomalyResult
from ..scoring.thresholds import SigmaThreshold
from ..signals.channels import SignalSchema
from ..signals.normalize import ZNormalizer

__all__ = ["BaseDetector"]


class BaseDetector(Detector):
    """Reusable detector skeleton implementing the :class:`Detector` contract."""

    name = "base"

    def __init__(
        self,
        threshold: Optional[ThresholdStrategy] = None,
        normalize: bool = True,
        min_anomaly_rate: float = 0.05,
        schema: Optional[SignalSchema] = None,
    ) -> None:
        self.threshold_strategy: ThresholdStrategy = threshold or SigmaThreshold()
        self._normalizer = ZNormalizer() if normalize else None
        self.min_anomaly_rate = float(min_anomaly_rate)
        self.schema = schema
        self._fitted = False
        self._baseline_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Hooks for subclasses
    # ------------------------------------------------------------------ #
    def _fit(self, X: np.ndarray) -> None:  # pragma: no cover - abstract-ish
        """Learn detector-specific state from normalized baseline ``X``."""
        raise NotImplementedError

    def _score(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Return per-row anomaly scores for normalized ``X``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared machinery
    # ------------------------------------------------------------------ #
    def _prepare(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self._normalizer is None:
            return X
        if fit:
            return self._normalizer.fit_transform(X)
        return self._normalizer.transform(X)

    def fit(self, X: np.ndarray) -> "BaseDetector":
        Xn = self._prepare(X, fit=True)
        if self.schema is None:
            self.schema = SignalSchema.generic(Xn.shape[1])
        self._fit(Xn)
        base_scores = np.asarray(self._score(Xn), dtype=float).ravel()
        self.threshold_strategy.fit(base_scores)
        self._baseline_scores = base_scores
        self._fitted = True
        return self

    def score(self, window: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError(f"{self.name} detector is not fitted")
        Xn = self._prepare(window)
        return np.asarray(self._score(Xn), dtype=float).ravel()

    def is_anomaly(self, window: np.ndarray) -> AnomalyResult:
        scores = self.score(window)
        thr = float(self.threshold_strategy.threshold(scores))
        mask = scores > thr
        rate = float(mask.mean()) if scores.size else 0.0
        aggregate = float(scores.max()) if scores.size else 0.0
        is_anom = rate >= self.min_anomaly_rate
        attribution = self._attribution(window, mask)
        return AnomalyResult(
            scores=scores,
            score=aggregate,
            threshold=thr,
            is_anomaly=is_anom,
            anomaly_rate=rate,
            attribution=attribution,
            detector=self.name,
            meta={"n_points": int(scores.size)},
        )

    def _attribution(self, window: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Per-channel contribution to the anomaly (normalized to sum 1).

        A generic, model-agnostic root-cause hint based on per-channel deviation
        over the flagged points. Detectors with a stronger causal signal can
        override this.
        """
        Xn = self._prepare(window)
        if mask.any():
            sub = Xn[mask]
        else:
            sub = Xn
        dev = np.abs(sub).mean(axis=0)
        total = dev.sum()
        if total <= 0:
            weights = np.ones_like(dev) / len(dev)
        else:
            weights = dev / total
        names = (
            self.schema.names
            if self.schema is not None
            else [f"ch{i}" for i in range(len(weights))]
        )
        return {name: float(w) for name, w in zip(names, weights)}
