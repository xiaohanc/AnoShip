"""Causal-inference detector.

Reference implementation of the core idea from:

    Xiangwei Chen, Hancheng Xiao, Zhixia Zeng, Shi Zhang, Ruliang Xiao.
    "Fine-Grained Multivariate Time Series Anomaly Detection via Causal
    Inference." Knowledge-Based Systems, 2025.
    https://www.sciencedirect.com/science/article/abs/pii/S0950705125018039

Core idea
---------
In a healthy system, channels obey stable inter-dependencies (one signal is
predictable from the others). The detector learns this dependency structure and
flags windows where a channel's observed value diverges from what its causal
parents predict. Because the residual is computed *per channel*, the detector
yields fine-grained, channel-level root-cause attribution -- pointing at the
likely source of a regression rather than just its symptom.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.registry import register_detector
from .base import BaseDetector

__all__ = ["CausalInferenceDetector"]


@register_detector("causal")
class CausalInferenceDetector(BaseDetector):
    name = "causal"

    def __init__(self, ridge: float = 1e-2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ridge = float(ridge)
        self._coefs: List[np.ndarray] = []
        self._resid_std: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        n, c = X.shape
        self._coefs = []
        residuals = np.zeros_like(X)
        for j in range(c):
            others = np.delete(X, j, axis=1)
            others_b = np.hstack([others, np.ones((n, 1))])  # bias term
            # Ridge-regularized least squares: predict channel j from the rest.
            a = others_b.T @ others_b + self.ridge * np.eye(others_b.shape[1])
            b = others_b.T @ X[:, j]
            w = np.linalg.solve(a, b)
            self._coefs.append(w)
            residuals[:, j] = X[:, j] - others_b @ w
        self._resid_std = residuals.std(axis=0) + 1e-8

    def _residuals(self, X: np.ndarray) -> np.ndarray:
        n, c = X.shape
        residuals = np.zeros_like(X)
        for j in range(c):
            others = np.delete(X, j, axis=1)
            others_b = np.hstack([others, np.ones((n, 1))])
            residuals[:, j] = X[:, j] - others_b @ self._coefs[j]
        return residuals / self._resid_std

    def _score(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] < 2:
            # No causal structure with a single channel: fall back to |z|.
            return np.abs(X).ravel()
        z = self._residuals(X)
        return np.sqrt((z**2).mean(axis=1))

    def _attribution(self, window: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        Xn = self._prepare(window)
        if Xn.shape[1] < 2:
            return super()._attribution(window, mask)
        z = np.abs(self._residuals(Xn))
        rows = z[mask] if mask.any() else z
        contrib = rows.mean(axis=0)
        total = contrib.sum()
        weights = contrib / total if total > 0 else np.ones_like(contrib) / len(contrib)
        names = (
            self.schema.names
            if self.schema is not None
            else [f"ch{i}" for i in range(len(weights))]
        )
        return {name: float(w) for name, w in zip(names, weights)}
