"""Multi-head Counterfactual Attention (the causal half of SCID's DCRE).

The factual representation forms the queries; the counterfactual representation
(variables stripped of temporal structure) forms the keys/values. The attention
context is what the factual variable would look like if explained purely by the
structureless counterfactual; the *difference* between the factual representation
and that context is the variable's genuine, structure-carried contribution.

A per-variable sigmoid gate scales this contribution back into the fused
representation (channel gating), and a normalized version of the gate is exposed
as a per-variable attribution weight.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

__all__ = ["CounterfactualAttention"]


class CounterfactualAttention(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 1)

    def forward(
        self, factual: torch.Tensor, counterfactual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross-attend factual queries over counterfactual keys/values.

        Parameters
        ----------
        factual, counterfactual:
            Per-variable representations ``(B, V, dim)``.

        Returns
        -------
        (fused, gate, var_weight, attn):
            ``fused`` ``(B, V, dim)`` gated fusion; ``gate`` ``(B, V)`` in
            ``(0, 1)``; ``var_weight`` ``(B, V)`` normalized to sum to 1 per
            sample; ``attn`` ``(B, V, V)`` with rows summing to 1.
        """
        Q = self.q(factual)
        K = self.k(counterfactual)
        Vv = self.v(counterfactual)
        attn = torch.softmax(torch.matmul(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        ctx = torch.matmul(attn, Vv)  # (B, V, dim)
        contrib = factual - ctx  # structure-carried deviation
        gate = torch.sigmoid(self.gate(contrib)).squeeze(-1)  # (B, V)
        var_weight = gate / gate.sum(dim=1, keepdim=True).clamp_min(self.eps)
        fused = factual + gate.unsqueeze(-1) * contrib
        return fused, gate, var_weight, attn
