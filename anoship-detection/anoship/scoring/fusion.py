"""Score-fusion utilities for combining multiple detectors.

Different detectors output scores on different scales, so fusion first puts each
score stream on a common footing (robust z-scaling) and then combines them.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["robust_zscale", "fuse_scores"]

_EPS = 1e-8


def robust_zscale(scores: np.ndarray) -> np.ndarray:
    """Scale scores by median / MAD so detectors become comparable."""
    scores = np.asarray(scores, dtype=float)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median)) + _EPS
    return (scores - median) / (1.4826 * mad)


def fuse_scores(
    score_streams: Sequence[np.ndarray],
    method: str = "mean",
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Combine multiple equal-length score streams into one.

    Parameters
    ----------
    score_streams:
        Iterable of 1-D arrays of identical length.
    method:
        One of ``"mean"``, ``"max"``, or ``"weighted"``.
    weights:
        Per-stream weights, required when ``method == "weighted"``.
    """
    if not score_streams:
        raise ValueError("no score streams to fuse")
    scaled = np.stack([robust_zscale(s) for s in score_streams], axis=0)
    if method == "mean":
        return scaled.mean(axis=0)
    if method == "max":
        return scaled.max(axis=0)
    if method == "weighted":
        if weights is None:
            raise ValueError("weights required for weighted fusion")
        w = np.asarray(weights, dtype=float)
        if len(w) != scaled.shape[0]:
            raise ValueError("number of weights must match number of streams")
        w = w / (w.sum() + _EPS)
        return (scaled * w[:, None]).sum(axis=0)
    raise ValueError(f"unknown fusion method: {method!r}")
