"""Anti-habituation similarity enhancement (AHSC, Eq. 8-9).

The fly olfactory circuit *habituates*: lateral-inhibition weights ``W`` learn the
recurring background so the response ``x = max(S - W, 0)`` to a familiar stimulus
fades over time. AHSC turns this into a similarity-enhancement preprocessing step:
by subtracting the learned common background from each incoming sparse code, the
shared component of same-cluster points is removed, raising their intra-cluster
cohesion, while a genuinely novel pattern keeps a large response.

Weight dynamics follow the habituation model the paper builds on:

    x_t = max(S_t - W_t, 0)                 (Eq. 8)
    W_{t+1} = (1 - beta) * W_t + alpha * x_t   (Eq. 9)

with habituation rate ``alpha`` and recovery rate ``beta`` in [0, 1]. (The printed
Eq. 9 has a transcribed sign; this is the stable habituation form it references,
whose fixed point suppresses but never erases a recurring background.)
"""

from __future__ import annotations

import numpy as np

__all__ = ["AntiHabituation"]


class AntiHabituation:
    """Streaming anti-habituation enhancement over a sequence of codes."""

    def __init__(self, dim: int, alpha: float = 0.1, beta: float = 0.05) -> None:
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.weights = np.zeros(self.dim, dtype=float)

    def reset(self) -> None:
        """Clear the learned background weights."""
        self.weights = np.zeros(self.dim, dtype=float)

    def step(self, s: np.ndarray) -> np.ndarray:
        """Enhance a single code ``s`` and update the habituation weights."""
        s = np.asarray(s, dtype=float)
        x = np.maximum(s - self.weights, 0.0)
        self.weights = (1.0 - self.beta) * self.weights + self.alpha * x
        return x

    def transform(self, S: np.ndarray) -> np.ndarray:
        """Enhance a block of codes row by row (single scan), keeping state."""
        S = np.asarray(S, dtype=float)
        if S.ndim == 1:
            return self.step(S)
        return np.vstack([self.step(row) for row in S])
