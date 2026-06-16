"""Graph Attention over variables (the spatial half of SCID's DCRE).

A multi-head GAT layer treats each variable as a node and attends over the other
variables, optionally restricted to the edges of the cached causal graph. Self
loops are always added so that every node has at least one valid neighbour (no
all-``-inf`` attention row).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

__all__ = ["GraphAttention"]


class GraphAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 2,
        dropout: float = 0.1,
        concat: bool = True,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    @property
    def output_dim(self) -> int:
        return self.heads * self.out_dim if self.concat else self.out_dim

    def forward(
        self, h: torch.Tensor, adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attend over variables.

        Parameters
        ----------
        h:
            Node features ``(B, V, in_dim)``.
        adj:
            Optional boolean adjacency ``(V, V)``; non-edges are masked out
            (self loops are always allowed).

        Returns
        -------
        (out, attn):
            ``out`` is ``(B, V, heads*out_dim)`` when ``concat`` else
            ``(B, V, out_dim)``; ``attn`` is ``(B, heads, V, V)`` with rows that
            sum to 1.
        """
        B, V, _ = h.shape
        Wh = self.W(h).view(B, V, self.heads, self.out_dim)  # (B, V, H, Do)
        e_src = (Wh * self.a_src).sum(-1)  # (B, V, H)
        e_dst = (Wh * self.a_dst).sum(-1)  # (B, V, H)
        # e[b,h,i,j] = leaky(src_i + dst_j)
        e = self.leaky(
            e_src.permute(0, 2, 1).unsqueeze(-1) + e_dst.permute(0, 2, 1).unsqueeze(-2)
        )  # (B, H, V, V)

        if adj is not None:
            eye = torch.eye(V, dtype=torch.bool, device=h.device)
            allowed = adj.bool() | eye  # always keep self loops
            e = e.masked_fill(~allowed.view(1, 1, V, V), float("-inf"))

        attn = torch.softmax(e, dim=-1)  # (B, H, V, V)
        attn = self.dropout(attn)
        Wh_h = Wh.permute(0, 2, 1, 3)  # (B, H, V, Do)
        out = torch.matmul(attn, Wh_h)  # (B, H, V, Do)
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(B, V, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)  # (B, V, Do)
        return out, attn
