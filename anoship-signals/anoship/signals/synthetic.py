"""Synthetic multivariate stream generator with injectable anomalies.

This lets the framework demonstrate end-to-end behavior with zero external data
and full reproducibility. The named, labelled anomaly patterns here are what the
scenario library composes into healthy / regression / drift / noisy runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["SyntheticStream"]


@dataclass
class SyntheticStream:
    """Generates correlated, seasonal multivariate streams + anomalies.

    Parameters
    ----------
    n_channels:
        Number of signal channels.
    seasonality:
        Period of the sinusoidal component (in steps).
    ar:
        AR(1) coefficient for temporally-correlated noise.
    noise:
        Standard deviation of the innovation noise.
    seed:
        RNG seed for reproducibility.
    """

    n_channels: int = 4
    seasonality: int = 50
    ar: float = 0.5
    noise: float = 0.15
    seed: int = 0

    def _rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def baseline(self, n: int, offset: int = 0) -> np.ndarray:
        """Generate ``n`` steps of nominal, anomaly-free behavior."""
        rng = self._rng(offset)
        t = np.arange(n)
        # Per-channel phase-shifted seasonality + shared latent factor.
        phases = np.linspace(0, np.pi, self.n_channels)
        season = np.sin(2 * np.pi * t[:, None] / self.seasonality + phases[None, :])
        latent = np.sin(2 * np.pi * t / (self.seasonality * 2.0))[:, None]
        X = season + 0.4 * latent

        # AR(1) correlated noise.
        eps = rng.normal(0.0, self.noise, size=(n, self.n_channels))
        e = np.zeros_like(eps)
        for i in range(1, n):
            e[i] = self.ar * e[i - 1] + eps[i]
        return X + e

    def _channels(self, channels: Optional[Sequence[int]]) -> np.ndarray:
        if channels is None:
            return np.arange(self.n_channels)
        return np.asarray(channels, dtype=int)

    def inject_spike(
        self,
        X: np.ndarray,
        start: int,
        length: int = 5,
        magnitude: float = 4.0,
        channels: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add a short, sharp burst -- an overt failure."""
        X = np.array(X, dtype=float, copy=True)
        labels = np.zeros(X.shape[0], dtype=int)
        cols = self._channels(channels)
        end = min(start + length, X.shape[0])
        X[start:end][:, cols] += magnitude
        labels[start:end] = 1
        return X, labels

    def inject_drift(
        self,
        X: np.ndarray,
        start: int,
        slope: float = 0.05,
        channels: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add a gradual distributional drift -- a slow, silent failure."""
        X = np.array(X, dtype=float, copy=True)
        labels = np.zeros(X.shape[0], dtype=int)
        cols = self._channels(channels)
        n = X.shape[0]
        ramp = np.arange(n - start) * slope
        X[start:][:, cols] += ramp[:, None]
        labels[start:] = 1
        return X, labels

    def inject_regression(
        self,
        X: np.ndarray,
        start: int,
        shift: float = 0.8,
        channels: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add a subtle persistent level shift -- a nuanced behavioral change.

        This mirrors the hardest real case: a model update whose regression
        presents as a small but sustained behavioral shift rather than a crash.
        """
        X = np.array(X, dtype=float, copy=True)
        labels = np.zeros(X.shape[0], dtype=int)
        cols = self._channels(channels)
        X[start:][:, cols] += shift
        labels[start:] = 1
        return X, labels

    def add_noise(
        self, X: np.ndarray, scale: float = 1.0, offset: int = 99
    ) -> np.ndarray:
        """Add heavy-tailed observation noise *without* a true anomaly.

        Used to test false-alarm robustness: a good detector should not gate on
        noise alone.
        """
        rng = self._rng(offset)
        bursts = rng.standard_t(df=3, size=X.shape) * (self.noise * scale)
        return np.array(X, dtype=float, copy=True) + bursts
