"""Dynamic Causal Representation Encoder (DCRE).

DCRE turns a window ``(B, L, V)`` into a causal, per-variable representation:

1. a **per-variable GRU** summarizes each variable's temporal dynamics
   (the window is reshaped ``(B, L, V) -> (B*V, L, 1)`` so one shared GRU runs
   over every variable independently),
2. a **GAT** layer mixes information across variables along the (optional) cached
   causal graph,
3. **Multi-head Counterfactual Attention** contrasts the factual representation
   against the counterfactual one and gates each variable by its genuine,
   structure-carried contribution.

It returns the per-variable channel gate (used by the PRP decoder to reweight
variables), a normalized per-variable attribution weight, and pooled factual /
counterfactual latents (used by the MMD objective).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import nn

from ..config import SCIDConfig
from .gat import GraphAttention
from .mca import CounterfactualAttention

__all__ = ["DCRE", "DCREOutput"]


class DCREOutput(NamedTuple):
    gate: torch.Tensor  # (B, V) channel gate in (0, 1)
    var_weight: torch.Tensor  # (B, V) attribution weights (sum to 1)
    z_fact: torch.Tensor  # (B, embed_dim) factual latent
    z_cf: torch.Tensor  # (B, embed_dim) counterfactual latent
    fused: torch.Tensor  # (B, V, embed_dim) gated per-variable repr


class DCRE(nn.Module):
    def __init__(self, config: SCIDConfig, n_vars: int) -> None:
        super().__init__()
        self.config = config
        self.n_vars = n_vars
        self.gru = nn.GRU(1, config.gru_hidden, batch_first=True)
        self.gat = GraphAttention(
            config.gru_hidden,
            config.gat_hidden,
            heads=config.gat_heads,
            dropout=config.dropout,
            concat=True,
        )
        self.proj = nn.Linear(self.gat.output_dim, config.embed_dim)
        self.mca = CounterfactualAttention(config.embed_dim)
        self.latent = nn.Linear(config.embed_dim, config.embed_dim)

    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Per-variable GRU summary: ``(B, L, V) -> (B, V, gru_hidden)``."""
        B, L, V = x.shape
        xv = x.permute(0, 2, 1).reshape(B * V, L, 1)
        _, h_n = self.gru(xv)  # h_n: (1, B*V, gru_hidden)
        return h_n[-1].reshape(B, V, self.config.gru_hidden)

    def forward(
        self,
        x: torch.Tensor,
        x_cf: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> DCREOutput:
        h = self._encode_nodes(x)
        h_cf = self._encode_nodes(x_cf)
        g, _ = self.gat(h, adj)
        g_cf, _ = self.gat(h_cf, adj)
        f = self.proj(g)  # (B, V, embed)
        f_cf = self.proj(g_cf)  # (B, V, embed)
        fused, gate, var_weight, _ = self.mca(f, f_cf)
        z_fact = self.latent(fused.mean(dim=1))
        z_cf = self.latent(f_cf.mean(dim=1))
        return DCREOutput(
            gate=gate, var_weight=var_weight, z_fact=z_fact, z_cf=z_cf, fused=fused
        )
