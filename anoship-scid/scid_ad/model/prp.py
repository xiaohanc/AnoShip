"""Predictive Reconstruction Process (PRP) decoder.

A single shared-parameter decoder is run twice over each window:

* the **reconstruction** pass uses a bidirectional self-attention with a key mask
  (fully-masked time steps are blocked), so the model in-fills masked entries
  from visible context, and
* the **prediction** pass uses a causal look-ahead mask so a position can attend
  only to itself and the past -- no future leakage.

Because both passes use the *same* weights, the decoder must learn a
representation consistent with both objectives. The two outputs are later
combined (``0.5 * (recon + pred)``) into the reconstruction target ``r``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from ..config import SCIDConfig
from ..masking import causal_lookahead_mask

__all__ = ["PRPDecoder", "TimeSelfAttention"]


class TimeSelfAttention(nn.Module):
    """Multi-head self-attention over the time axis with an additive mask."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("decoder_hidden must be divisible by the head count")
        self.heads = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, add_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.dk)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, dk)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dk**0.5)
        if add_mask is not None:
            scores = scores + add_mask  # broadcasts over batch/heads/query
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)  # (B, H, L, dk)
        ctx = ctx.transpose(1, 2).reshape(B, L, D)
        return self.out(ctx)


class PRPDecoder(nn.Module):
    def __init__(self, config: SCIDConfig, n_vars: int, heads: int = 2) -> None:
        super().__init__()
        d = config.decoder_hidden
        self.input_proj = nn.Linear(n_vars, d)
        self.attn = TimeSelfAttention(d, heads)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * d, d),
        )
        self.out_proj = nn.Linear(d, n_vars)

    def _pass(
        self, tokens: torch.Tensor, add_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        h = self.norm1(tokens + self.attn(tokens, add_mask))
        h = self.norm2(h + self.ff(h))
        return self.out_proj(h)

    def reconstruction_pass(
        self, x_input: torch.Tensor, key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Bidirectional in-filling pass. ``key_mask`` is ``(B, 1, L)`` additive."""
        tokens = self.input_proj(x_input)
        mask = None
        if key_mask is not None:
            mask = key_mask.unsqueeze(1)  # (B, 1, 1, L) -> broadcast over heads/query
        return self._pass(tokens, mask)

    def prediction_pass(self, x_input: torch.Tensor) -> torch.Tensor:
        """Causal pass: position ``i`` attends only to ``<= i`` (no future leak)."""
        tokens = self.input_proj(x_input)
        L = x_input.shape[1]
        causal = causal_lookahead_mask(L, device=x_input.device)
        return self._pass(tokens, causal)

    def forward(
        self,
        recon_input: torch.Tensor,
        pred_input: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        recon = self.reconstruction_pass(recon_input, key_mask)
        pred = self.prediction_pass(pred_input)
        return recon, pred
