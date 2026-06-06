"""Normalization helpers used to stabilize anomaly scoring across channels."""

from __future__ import annotations

import numpy as np

from ..core.errors import NotFittedError

__all__ = ["ZNormalizer", "RobustNormalizer"]

_EPS = 1e-8


class ZNormalizer:
    """Standard per-channel z-normalization (mean / std)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZNormalizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + _EPS
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise NotFittedError("ZNormalizer must be fitted before transform")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class RobustNormalizer:
    """Median / IQR normalization, resilient to outliers in the baseline.

    Robust scaling matters for deployment safety: a few contaminated points in
    the baseline should not desensitize the detector to genuine regressions.
    """

    def __init__(self) -> None:
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustNormalizer":
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        self.iqr_ = (q75 - q25) + _EPS
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.iqr_ is None:
            raise NotFittedError("RobustNormalizer must be fitted before transform")
        return (np.asarray(X, dtype=float) - self.median_) / self.iqr_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
