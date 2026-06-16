"""Maximum Mean Discrepancy (MMD) with a mixture-of-Gaussian kernel.

SCID's counterfactual-differentiation objective uses MMD to (a) *pull* the
factual representation toward the original/factual distribution and (b) *push* it
away from the counterfactual distribution. A learnable weight balances the two
pulls so the model can adapt how strongly it separates factual from
counterfactual.

The multi-bandwidth Gaussian kernel makes the statistic sensitive across several
length scales without hand-tuning a single bandwidth.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

__all__ = ["gaussian_mmd2", "MMDLoss"]


def _mixture_gram(
    a: torch.Tensor, b: torch.Tensor, bandwidths: Sequence[float]
) -> torch.Tensor:
    """Mean Gaussian kernel matrix over a mixture of bandwidths, ``(Na, Nb)``."""
    d2 = torch.cdist(a, b, p=2.0) ** 2
    k = torch.zeros_like(d2)
    for bw in bandwidths:
        k = k + torch.exp(-d2 / (2.0 * float(bw) ** 2))
    return k / float(len(bandwidths))


def gaussian_mmd2(
    x: torch.Tensor, y: torch.Tensor, bandwidths: Sequence[float]
) -> torch.Tensor:
    """Biased MMD^2 between sample sets ``x`` and ``y``.

    Parameters
    ----------
    x, y:
        Sample sets of shape ``(Nx, d)`` and ``(Ny, d)``.

    Returns
    -------
    Scalar MMD^2. It is ``0`` when ``x`` and ``y`` are identical sample sets and
    grows as their distributions differ.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("gaussian_mmd2 expects (N, d) sample sets")
    kxx = _mixture_gram(x, x, bandwidths).mean()
    kyy = _mixture_gram(y, y, bandwidths).mean()
    kxy = _mixture_gram(x, y, bandwidths).mean()
    return kxx + kyy - 2.0 * kxy


class MMDLoss(nn.Module):
    """Pull-to-factual / push-from-counterfactual MMD with a learnable balance.

    ``loss = w * MMD^2(z, z_pos) - (1 - w) * MMD^2(z, z_neg)`` where ``w =
    sigmoid(weight)`` is learned. Minimizing pulls the representation ``z`` toward
    the factual reference ``z_pos`` while pushing it away from the counterfactual
    reference ``z_neg``.
    """

    def __init__(self, bandwidths: Sequence[float]) -> None:
        super().__init__()
        self.bandwidths = list(bandwidths)
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(
        self, z: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor
    ) -> torch.Tensor:
        w = torch.sigmoid(self.weight)
        pull = gaussian_mmd2(z, z_pos, self.bandwidths)
        push = gaussian_mmd2(z, z_neg, self.bandwidths)
        return w * pull - (1.0 - w) * push
