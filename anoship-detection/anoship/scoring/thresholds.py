"""Thresholding strategies that turn anomaly scores into binary decisions.

Separating *scoring* (a detector's job) from *thresholding* lets the same
detector be tuned for different operating points -- e.g. tighter sensitivity for
higher-frequency model updates -- without changing the detector itself.
"""

from __future__ import annotations

import numpy as np

from ..core.interfaces import ThresholdStrategy
from ..core.registry import register_threshold

__all__ = [
    "StaticThreshold",
    "SigmaThreshold",
    "QuantileThreshold",
    "AdaptiveThreshold",
]


@register_threshold("static")
class StaticThreshold(ThresholdStrategy):
    """A fixed threshold, independent of the data."""

    def __init__(self, value: float = 3.0) -> None:
        self.value = float(value)

    def fit(self, scores: np.ndarray) -> "StaticThreshold":
        return self

    def threshold(self, scores: np.ndarray) -> float:
        return self.value


@register_threshold("sigma")
class SigmaThreshold(ThresholdStrategy):
    """``mean + k * std`` calibrated on baseline scores (3-sigma by default)."""

    def __init__(self, k: float = 3.0) -> None:
        self.k = float(k)
        self._mean = 0.0
        self._std = 1.0

    def fit(self, scores: np.ndarray) -> "SigmaThreshold":
        scores = np.asarray(scores, dtype=float)
        self._mean = float(scores.mean())
        self._std = float(scores.std() + 1e-8)
        return self

    def threshold(self, scores: np.ndarray) -> float:
        return self._mean + self.k * self._std


@register_threshold("quantile")
class QuantileThreshold(ThresholdStrategy):
    """Threshold at a high quantile of the baseline score distribution."""

    def __init__(self, q: float = 0.99) -> None:
        if not 0.0 < q < 1.0:
            raise ValueError("q must be in (0, 1)")
        self.q = float(q)
        self._value = 0.0

    def fit(self, scores: np.ndarray) -> "QuantileThreshold":
        scores = np.asarray(scores, dtype=float)
        self._value = float(np.quantile(scores, self.q))
        return self

    def threshold(self, scores: np.ndarray) -> float:
        return self._value


@register_threshold("adaptive")
class AdaptiveThreshold(ThresholdStrategy):
    """Self-adjusting threshold combining a baseline floor with a live EWMA.

    The threshold tracks the recent score level (EWMA) plus a sigma margin, but
    never drops below the baseline-calibrated floor. This supports
    high-frequency deployment, where a static threshold either over- or
    under-fires as the score distribution shifts between updates.
    """

    def __init__(self, k: float = 3.0, alpha: float = 0.1) -> None:
        self.k = float(k)
        self.alpha = float(alpha)
        self._floor = 0.0
        self._std = 1.0

    def fit(self, scores: np.ndarray) -> "AdaptiveThreshold":
        scores = np.asarray(scores, dtype=float)
        self._floor = float(scores.mean() + self.k * (scores.std() + 1e-8))
        self._std = float(scores.std() + 1e-8)
        return self

    def threshold(self, scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return self._floor
        # EWMA of the current window's level.
        weights = (1 - self.alpha) ** np.arange(scores.size)[::-1]
        ewma = float(np.average(scores, weights=weights))
        return max(self._floor, ewma + self.k * self._std)
