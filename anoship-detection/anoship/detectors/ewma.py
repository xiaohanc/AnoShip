"""EWMA residual detector -- a lightweight baseline.

Not tied to a publication; included as a fast, transparent reference point so
the published detectors can be compared against a simple control.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_detector
from .base import BaseDetector

__all__ = ["EWMAResidualDetector"]


@register_detector("ewma")
class EWMAResidualDetector(BaseDetector):
    """Scores the residual between each observation and its EWMA forecast."""

    name = "ewma"

    def __init__(self, alpha: float = 0.3, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)

    def _fit(self, X: np.ndarray) -> None:
        # Stateless beyond normalization; nothing to learn.
        self._seed = X[0].copy()

    def _score(self, X: np.ndarray) -> np.ndarray:
        pred = np.empty_like(X)
        pred[0] = X[0]
        a = self.alpha
        for i in range(1, len(X)):
            pred[i] = a * X[i - 1] + (1 - a) * pred[i - 1]
        residual = np.abs(X - pred)
        return residual.mean(axis=1)
