"""Differentiable soft-DTW, implemented in-repo (no external dependency).

soft-DTW (Cuturi & Blondel, 2017) replaces the hard ``min`` in the dynamic-time-
warping recursion with a smooth ``soft-min`` (a ``-gamma * logsumexp``), making
the alignment cost differentiable. SCID uses it as a "granularity adjustment"
objective so that reconstructions match the *shape* of the signal, tolerant to
small temporal shifts.

The recursion is ``O(L * M)`` per pair; SCID's windows are short (``L`` ~ 30), so
a plain Python DP loop over the (tiny) time axis is fast enough and keeps the
implementation transparent and dependency-free.
"""

from __future__ import annotations

import torch

__all__ = ["soft_dtw", "soft_dtw_normalized"]


def _squared_cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances, shape ``(B, L, M)``."""
    return torch.cdist(x, y, p=2.0) ** 2


def soft_dtw(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Batched soft-DTW discrepancy between two sequences.

    Parameters
    ----------
    x, y:
        Sequences of shape ``(B, L, d)`` and ``(B, M, d)``.
    gamma:
        Smoothing temperature. As ``gamma -> 0`` the soft-min approaches the hard
        ``min`` and soft-DTW approaches classical DTW.

    Returns
    -------
    Tensor of shape ``(B,)`` with the soft-DTW value per batch element.
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("soft_dtw expects (B, L, d) inputs")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    D = _squared_cdist(x, y)  # (B, L, M)
    B, L, M = D.shape
    inf = torch.full((B,), float("inf"), device=x.device, dtype=x.dtype)
    zero = torch.zeros((B,), device=x.device, dtype=x.dtype)

    # R is an (L+1, M+1) grid of (B,) tensors; build functionally to keep
    # autograd happy (no in-place writes into a grad-tracked tensor).
    R = [[inf for _ in range(M + 1)] for _ in range(L + 1)]
    R[0][0] = zero
    for i in range(1, L + 1):
        for j in range(1, M + 1):
            stacked = torch.stack([R[i - 1][j - 1], R[i - 1][j], R[i][j - 1]], dim=0)
            soft_min = -gamma * torch.logsumexp(-stacked / gamma, dim=0)
            R[i][j] = D[:, i - 1, j - 1] + soft_min
    return R[L][M]


def soft_dtw_normalized(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1
) -> torch.Tensor:
    """Mean soft-DTW per time step, averaged over the batch (a stable scalar).

    Dividing by the sequence length keeps the term on the same scale as the
    pointwise reconstruction loss regardless of window size.
    """
    L = x.shape[1]
    return soft_dtw(x, y, gamma).mean() / float(L)
