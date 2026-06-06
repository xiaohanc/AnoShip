"""Spatiotemporal-fusion detector.

Reference implementation of the core idea from:

    Weikang Shi, Hancheng Xiao, Zhipeng Qiu, Zhixia Zeng, Weifu Zhu, Shi Zhang,
    Ruliang Xiao. "MSTDF-AD: Modeling Spatiotemporal Dependency Fusion for
    Non-Stationary Time Series Anomaly Detection." Information Processing &
    Management, 2026.
    https://www.sciencedirect.com/science/article/pii/S0306457326002797

Core idea
---------
Fuse three complementary dependency views into one feature representation:

* **level**     -- the (locally detrended) value, handling non-stationarity,
* **temporal**  -- the step-to-step change, capturing temporal dependency,
* **spatial**   -- each channel's deviation from the cross-channel consensus.

Anomalies are scored as a standardized distance of the fused representation from
the baseline distribution, so a fault that is invisible in any single view but
visible in their *fusion* is still caught.

A full PyTorch implementation of this model lives in the companion ``MSTDF-AD``
repository and can be plugged into this same interface via
``anoship.adapters.torch_mstdf``.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_detector
from .base import BaseDetector

__all__ = ["SpatioTemporalFusionDetector"]


def _detrend(X: np.ndarray, window: int) -> np.ndarray:
    """Remove a local moving-average trend to handle non-stationarity."""
    if window <= 1:
        return X - X.mean(axis=0)
    kernel = np.ones(window) / window
    trend = np.empty_like(X)
    for j in range(X.shape[1]):
        trend[:, j] = np.convolve(X[:, j], kernel, mode="same")
    return X - trend


def _smooth(X: np.ndarray, sigma: float) -> np.ndarray:
    """Light Gaussian denoising so the temporal term is not noise-dominated."""
    if sigma <= 0:
        return X
    radius = max(1, int(3 * sigma))
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = np.convolve(X[:, j], kernel, mode="same")
    return out


@register_detector("spatiotemporal")
class SpatioTemporalFusionDetector(BaseDetector):
    name = "spatiotemporal"

    def __init__(
        self, detrend_window: int = 5, smooth_sigma: float = 1.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.detrend_window = int(detrend_window)
        self.smooth_sigma = float(smooth_sigma)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _fuse(self, X: np.ndarray) -> np.ndarray:
        Xs = _smooth(X, self.smooth_sigma)
        level = _detrend(Xs, self.detrend_window)
        temporal = np.vstack([np.zeros((1, Xs.shape[1])), np.diff(Xs, axis=0)])
        spatial = Xs - Xs.mean(axis=1, keepdims=True)
        return np.hstack([level, temporal, spatial])

    def _fit(self, X: np.ndarray) -> None:
        fused = self._fuse(X)
        self._mean = fused.mean(axis=0)
        self._std = fused.std(axis=0) + 1e-8

    def _score(self, X: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        z = (self._fuse(X) - self._mean) / self._std
        return np.sqrt((z**2).mean(axis=1))
