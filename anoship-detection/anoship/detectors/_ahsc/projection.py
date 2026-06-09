"""Drosophila olfactory projection + winner-take-all (AHSC, Eq. 5-7).

The fruit-fly olfactory circuit expands the input into a much higher-dimensional
space through a *sparse binary* random projection (PN -> KC), then applies a
"winner-takes-all" inhibition (APL) that keeps only the strongest activations and
zeroes the rest. This sparse, expanded code is a locality-sensitive hash that
(a) preserves the relative geometry of the input while (b) sharply mitigating the
"curse of dimensionality" -- the role this stage plays in AHSC's preprocessing.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FlyProjection", "winner_take_all"]


class FlyProjection:
    """Sparse binary random projection ``Y = M . X`` (Eq. 5-6).

    Parameters
    ----------
    input_dim:
        Dimension ``d`` of a single stream point.
    expansion:
        Expansion factor; the projection has ``m = expansion * d`` rows
        (the paper uses ``m`` roughly 40x ``d``).
    density:
        Bernoulli connection probability for each entry of ``M``.
    seed:
        Seed for the (fixed) projection matrix.
    """

    def __init__(
        self,
        input_dim: int,
        expansion: int = 40,
        density: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.input_dim = int(input_dim)
        self.expansion = int(expansion)
        self.density = float(density)
        self.seed = int(seed)
        self.output_dim = self.expansion * self.input_dim

        rng = np.random.default_rng(self.seed)
        M = (rng.random((self.output_dim, self.input_dim)) < self.density).astype(float)
        # Guarantee every Kenyon cell samples at least one projection neuron.
        empty = np.where(M.sum(axis=1) == 0)[0]
        if empty.size:
            cols = rng.integers(0, self.input_dim, size=empty.size)
            M[empty, cols] = 1.0
        self.matrix = M

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project ``X`` of shape ``(n, d)`` to ``(n, m)`` via ``X . M^T``."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.matrix.T


def winner_take_all(Y: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Keep the top ``frac`` activations per row, zero the rest (Eq. 7).

    At least one winner is always retained, so an all-equal row still yields a
    valid sparse code.
    """
    Y = np.asarray(Y, dtype=float)
    single = Y.ndim == 1
    if single:
        Y = Y.reshape(1, -1)
    n, m = Y.shape
    k = max(1, int(round(frac * m)))
    k = min(k, m)
    out = np.zeros_like(Y)
    # Indices of the k largest activations per row.
    winners = np.argpartition(Y, m - k, axis=1)[:, m - k:]
    rows = np.arange(n)[:, None]
    out[rows, winners] = Y[rows, winners]
    return out[0] if single else out
