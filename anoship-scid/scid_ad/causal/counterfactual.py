"""Counterfactual window construction for SCID's causal reasoning.

SCID contrasts the *factual* window against a *counterfactual* one in which the
temporal structure of every variable is removed: each variable is replaced by
its own per-variable temporal mean over the window (all variables
simultaneously). The Multi-head Counterfactual Attention then differentiates the
factual representation from this structureless reference.
"""

from __future__ import annotations

import torch

__all__ = ["counterfactual"]


def counterfactual(x: torch.Tensor) -> torch.Tensor:
    """Return the counterfactual of window ``x``.

    Each ``(t, v)`` entry is replaced by the temporal mean of variable ``v`` over
    the window, so every variable becomes constant in time while keeping its
    average level.

    Parameters
    ----------
    x:
        Factual window of shape ``(B, L, V)``.

    Returns
    -------
    Counterfactual window of shape ``(B, L, V)`` (constant along the time axis).
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B, L, V) input, got shape {tuple(x.shape)}")
    mean = x.mean(dim=1, keepdim=True)  # (B, 1, V)
    return mean.expand_as(x).contiguous()
