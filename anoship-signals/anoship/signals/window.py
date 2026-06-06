"""Sliding-window utilities for streaming evaluation.

Production deployment-safety evaluation happens over *windows* of a continuous
stream rather than over a fixed dataset. These helpers turn an array or a live
feed of rows into overlapping windows the detectors can score.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, List, Optional

import numpy as np

__all__ = ["SlidingWindow", "StreamBuffer"]


class SlidingWindow:
    """Splits a ``(T, C)`` array into overlapping windows of fixed size."""

    def __init__(self, size: int, stride: int = 1) -> None:
        if size <= 0:
            raise ValueError("window size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.size = size
        self.stride = stride

    def windows(self, X: np.ndarray) -> Iterator[np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        if n < self.size:
            return
        for start in range(0, n - self.size + 1, self.stride):
            yield X[start : start + self.size]

    def as_list(self, X: np.ndarray) -> List[np.ndarray]:
        return list(self.windows(X))


class StreamBuffer:
    """A bounded ring buffer for online, row-by-row stream evaluation."""

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = maxlen
        self._buf: Deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, row: np.ndarray) -> None:
        self._buf.append(np.asarray(row, dtype=float).ravel())

    def extend(self, rows: np.ndarray) -> None:
        for row in np.asarray(rows, dtype=float):
            self.push(row)

    @property
    def full(self) -> bool:
        return len(self._buf) == self.maxlen

    def window(self) -> Optional[np.ndarray]:
        """Return the current window, or ``None`` if not yet full."""
        if not self._buf:
            return None
        return np.vstack(list(self._buf))

    def __len__(self) -> int:
        return len(self._buf)
