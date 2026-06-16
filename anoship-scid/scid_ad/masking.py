"""Adaptive masking for SCID's Predictive Reconstruction Process (PRP).

SCID's PRP runs a shared decoder twice over each window:

* a **reconstruction** pass that masks part of the input (so the model must
  in-fill the masked entries from the visible context), and
* a **prediction** pass that uses a causal look-ahead mask (so a position may
  only attend to itself and the past).

The masking *rate* is adaptive: variables with higher coefficient of variation
(CV) are noisier/more informative and get masked more aggressively, up to a cap.
The CV is computed on the raw training split (before normalization) and cached.

All attention masks here are *additive* masks (``0`` keep, ``-inf`` block) shaped
to broadcast against attention logits ``(..., L_query, L_key)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "coefficient_of_variation",
    "mask_rate",
    "apply_input_mask",
    "recon_key_mask",
    "causal_lookahead_mask",
    "NEG_INF",
]

NEG_INF = float("-inf")


def coefficient_of_variation(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-variable coefficient of variation ``std / |mean|``.

    Computed on the *raw* training split (before normalization). The mean is
    floored in magnitude by ``eps`` to guard the near-zero-mean edge case.

    Parameters
    ----------
    X:
        Tensor of shape ``(T, V)`` (or ``(B, L, V)``; flattened over all but the
        last axis).
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False)
    denom = mean.abs().clamp_min(eps)
    return std / denom


def mask_rate(cv: torch.Tensor, eta: float = 0.3, cap: float = 0.5) -> torch.Tensor:
    """Per-variable masking rate ``clamp(eta * CV, 0, cap)``."""
    return (eta * cv).clamp(0.0, cap)


def apply_input_mask(
    x: torch.Tensor,
    rate: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly zero out entries of ``x`` at the per-variable ``rate``.

    Parameters
    ----------
    x:
        Input window ``(B, L, V)``.
    rate:
        Per-variable masking probability ``(V,)``.

    Returns
    -------
    (x_masked, mask):
        ``x_masked`` with masked entries set to zero, and a boolean ``mask`` of
        shape ``(B, L, V)`` that is ``True`` exactly where entries were masked.
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B, L, V) input, got shape {tuple(x.shape)}")
    B, L, V = x.shape
    r = rate.to(x.device, x.dtype).reshape(1, 1, V)
    u = torch.rand((B, L, V), generator=generator, device=x.device, dtype=x.dtype)
    mask = u < r
    x_masked = x.masked_fill(mask, 0.0)
    return x_masked, mask


def recon_key_mask(mask: torch.Tensor) -> torch.Tensor:
    """Additive key mask for the reconstruction pass.

    A *time step* is treated as a blocked key when **all** of its variables are
    masked ("fully-masked keys set to -inf"), so queries reconstruct it from
    visible context. To avoid an all-``-inf`` (NaN) attention row, a time step is
    never blocked if it is fully masked across the entire window.

    Parameters
    ----------
    mask:
        Boolean input mask ``(B, L, V)`` from :func:`apply_input_mask`.

    Returns
    -------
    additive mask of shape ``(B, 1, L)`` broadcastable over query positions.
    """
    if mask.dim() != 3:
        raise ValueError(f"expected (B, L, V) mask, got shape {tuple(mask.shape)}")
    fully_masked = mask.all(dim=-1)  # (B, L)
    # Guard: if every key in a window is fully masked, keep them all (no -inf).
    all_blocked = fully_masked.all(dim=1, keepdim=True)  # (B, 1)
    fully_masked = fully_masked & ~all_blocked
    add = torch.zeros_like(fully_masked, dtype=torch.float32)
    add = add.masked_fill(fully_masked, NEG_INF)
    return add.unsqueeze(1)  # (B, 1, L)


def causal_lookahead_mask(
    length: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Additive causal mask ``(L, L)``: ``-inf`` strictly above the diagonal.

    Position ``i`` (query) may attend only to keys ``j <= i`` (itself and the
    past); future keys ``j > i`` are blocked.
    """
    full = torch.full((length, length), NEG_INF, device=device)
    return torch.triu(full, diagonal=1)
