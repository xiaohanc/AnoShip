"""scikit-learn adapter: plug any sklearn outlier estimator into anoship.

Demonstrates that the detector interface is genuinely open: an
``IsolationForest`` or ``OneClassSVM`` participates in the exact same health-gate
machinery as the built-in published detectors. Requires the optional
``sklearn`` extra.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_detector
from ..detectors.base import BaseDetector

__all__ = ["SklearnDetector"]


@register_detector("sklearn")
class SklearnDetector(BaseDetector):
    """Wraps a scikit-learn outlier/novelty estimator as a Detector."""

    name = "sklearn"

    def __init__(self, estimator=None, **kwargs) -> None:
        super().__init__(**kwargs)
        if estimator is None:
            try:
                from sklearn.ensemble import IsolationForest
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "scikit-learn is required for SklearnDetector; install "
                    "with 'pip install anoship[sklearn]'"
                ) from exc
            estimator = IsolationForest(random_state=0)
        self.estimator = estimator

    def _fit(self, X: np.ndarray) -> None:
        self.estimator.fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "score_samples"):
            return -np.asarray(self.estimator.score_samples(X), dtype=float)
        if hasattr(self.estimator, "decision_function"):
            return -np.asarray(self.estimator.decision_function(X), dtype=float)
        # Fall back to hard predictions mapped to {0, 1}.
        pred = np.asarray(self.estimator.predict(X), dtype=float)
        return (pred < 0).astype(float)
