"""The composed SCID model: DCRE encoder + PRP decoder.

``forward`` runs the encoder to obtain the per-variable channel gate and the
factual/counterfactual latents, gates the input, and runs the PRP decoder's two
passes. During training a CV-adaptive input mask is applied to the
reconstruction pass (the model must in-fill the masked entries); at inference the
input is left intact.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import nn

from ..causal.counterfactual import counterfactual
from ..config import SCIDConfig
from ..masking import apply_input_mask, recon_key_mask
from .dcre import DCRE
from .prp import PRPDecoder

__all__ = ["SCIDModel", "SCIDOutput"]


class SCIDOutput(NamedTuple):
    recon: torch.Tensor  # (B, L, V) reconstruction-pass output
    pred: torch.Tensor  # (B, L, V) prediction-pass output
    combined: torch.Tensor  # (B, L, V) r = 0.5 * (recon + pred)
    z_fact: torch.Tensor  # (B, embed_dim) factual latent
    z_cf: torch.Tensor  # (B, embed_dim) counterfactual latent
    var_weight: torch.Tensor  # (B, V) per-variable attribution weights
    gate: torch.Tensor  # (B, V) channel gate


class SCIDModel(nn.Module):
    def __init__(self, config: SCIDConfig, n_vars: int, prp_heads: int = 2) -> None:
        super().__init__()
        self.config = config
        self.n_vars = n_vars
        self.dcre = DCRE(config, n_vars)
        self.prp = PRPDecoder(config, n_vars, heads=prp_heads)

    def forward(
        self,
        x: torch.Tensor,
        rate: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> SCIDOutput:
        x_cf = counterfactual(x)
        dout = self.dcre(x, x_cf, adj)
        gate = dout.gate.unsqueeze(1)  # (B, 1, V) broadcast over time

        if rate is not None:
            x_masked, mask = apply_input_mask(x, rate, generator=generator)
            key_mask = recon_key_mask(mask)
        else:
            x_masked, key_mask = x, None

        recon_input = x_masked * gate
        pred_input = x * gate
        recon, pred = self.prp(recon_input, pred_input, key_mask)
        combined = 0.5 * (recon + pred)
        return SCIDOutput(
            recon=recon,
            pred=pred,
            combined=combined,
            z_fact=dout.z_fact,
            z_cf=dout.z_cf,
            var_weight=dout.var_weight,
            gate=dout.gate,
        )
