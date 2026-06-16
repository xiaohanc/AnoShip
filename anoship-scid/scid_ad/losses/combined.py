"""SCID's dual-objective loss with two-phase composition.

``L_total = alpha * L_PRP + beta * L_DTW + gamma * L_MMD`` where

* ``L_PRP`` is the Predictive Reconstruction Process loss (MSE of both the
  reconstruction and prediction passes against the original window),
* ``L_DTW`` is the soft-DTW granularity-adjustment term, and
* ``L_MMD`` is the counterfactual-differentiation term.

During the **warm-up** phase only ``L_DTW`` and ``L_MMD`` are active (the
representation/causal objectives are learned first); the **full** phase adds
``L_PRP`` so the decoder is trained against a well-formed representation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..config import SCIDConfig
from .mmd import MMDLoss
from .soft_dtw import soft_dtw_normalized

__all__ = ["SCIDLoss"]


class SCIDLoss(nn.Module):
    """Composite SCID objective (see module docstring)."""

    def __init__(self, config: SCIDConfig) -> None:
        super().__init__()
        self.config = config
        self.mmd = MMDLoss(config.mmd_bandwidths)

    def prp_loss(
        self, recon: torch.Tensor, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(recon, target) + F.mse_loss(pred, target)

    def dtw_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return soft_dtw_normalized(recon, target, self.config.dtw_gamma)

    def forward(
        self,
        *,
        recon: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        z: torch.Tensor,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
        warmup: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_prp = self.prp_loss(recon, pred, target)
        l_dtw = self.dtw_loss(recon, target)
        l_mmd = self.mmd(z, z_pos, z_neg)

        cfg = self.config
        if warmup:
            total = cfg.beta * l_dtw + cfg.gamma * l_mmd
        else:
            total = cfg.alpha * l_prp + cfg.beta * l_dtw + cfg.gamma * l_mmd

        parts = {
            "prp": float(l_prp.detach()),
            "dtw": float(l_dtw.detach()),
            "mmd": float(l_mmd.detach()),
            "total": float(total.detach()),
        }
        return total, parts
