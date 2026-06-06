"""Diffusion-denoising detector.

Reference implementation of the core ideas from two works:

    Xiaorui Huang, Hancheng Xiao, et al. "Toward robust anomaly detection in
    noisy time series via diffusion-driven denoising and disentanglement."
    The Journal of Supercomputing, 2026.
    https://link.springer.com/article/10.1007/s11227-026-08403-x

    Jiacai Chen, Hancheng Xiao, et al. "Diffusion-step attention consistency for
    multivariate time series anomaly detection." Knowledge-Based Systems, 2026.
    https://www.sciencedirect.com/science/article/pii/S0950705126004569

Core idea
---------
Project the signal onto a low-rank "normal" subspace, then evaluate the
reconstruction error at multiple denoising scales (a discrete analogue of
diffusion steps). Transient observation noise is washed out as denoising
strengthens, whereas a genuine anomaly leaves a reconstruction error that is
*consistent across steps*. Scoring on the error that survives the strongest
denoising disentangles real anomalies from noise -- the central robustness claim
of these papers.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_detector
from .base import BaseDetector

__all__ = ["DiffusionDenoiseDetector"]


def _gaussian_smooth(X: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth each channel along time with a small Gaussian kernel."""
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


@register_detector("diffusion")
class DiffusionDenoiseDetector(BaseDetector):
    name = "diffusion"

    def __init__(
        self,
        energy: float = 0.9,
        steps: tuple = (0.0, 1.0, 2.0, 4.0),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.energy = float(energy)
        self.steps = tuple(float(s) for s in steps)
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        # Low-rank "normal" subspace via SVD.
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        energy = np.cumsum(s**2) / np.sum(s**2)
        k = int(np.searchsorted(energy, self.energy) + 1)
        k = max(1, min(k, vt.shape[0]))
        self._components = vt[:k]  # (k, C)

    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._components is not None
        Xc = X - self._mean
        proj = Xc @ self._components.T @ self._components  # reconstruction
        return np.linalg.norm(Xc - proj, axis=1)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # Reconstruction error across denoising steps; keep the error that
        # persists under the strongest denoising (consistency across steps).
        errors = []
        for sigma in self.steps:
            Xs = _gaussian_smooth(X, sigma)
            errors.append(self._recon_error(Xs))
        errors = np.stack(errors, axis=1)  # (T, n_steps)
        # The min across steps == the error consistently present at every
        # denoising level: noise-robust, anomaly-sensitive.
        return errors.min(axis=1)
