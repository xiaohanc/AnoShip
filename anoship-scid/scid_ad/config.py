"""Configuration for the from-paper SCID model.

Two presets are provided:

* :meth:`SCIDConfig.quick` (the default used by the detector) -- small dims and
  few epochs so the model fits in seconds for tests and interactive use.
* :meth:`SCIDConfig.paper` -- a larger, paper-scale configuration. Note that
  even this preset is *not* a reproduction of the paper's numbers (no upstream
  code, no benchmark datasets bundled); it merely uses more representative
  capacities and schedules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SCIDConfig:
    """Hyper-parameters for the SCID model and its two-phase trainer."""

    # -- windowing / data --
    window_size: int = 30  # L; soft-DTW is O(L^2) so keep this small
    step_size: int = 10  # stride between overlapping windows

    # -- DCRE encoder --
    gru_hidden: int = 16  # per-variable GRU hidden size
    gat_hidden: int = 16  # GAT output size per variable
    gat_heads: int = 2  # GAT attention heads
    embed_dim: int = 16  # MCA / latent representation width
    dropout: float = 0.1

    # -- PRP decoder --
    decoder_hidden: int = 16  # shared-parameter decoder hidden size

    # -- masking --
    mask_eta: float = 0.3  # rate = clamp(eta * CV(X), 0, mask_cap)
    mask_cap: float = 0.5  # maximum masking rate

    # -- losses --
    alpha: float = 1.0  # weight on L_PRP (reconstruction/prediction)
    beta: float = 0.1  # weight on L_DTW (soft-DTW granularity)
    gamma: float = 0.1  # weight on L_MMD (counterfactual differentiation)
    dtw_gamma: float = 0.1  # soft-DTW smoothing temperature
    mmd_bandwidths: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])

    # -- causal graph (KSG-CMI) --
    cmi_k: int = 5  # k for the KSG k-NN CMI estimator
    cmi_top_k: int = 10  # cap conditioning set to top-k correlated vars
    causal_heads: int = 1  # number of lag/head terms h in S_ij
    causal_lambda: List[float] = field(default_factory=lambda: [1.0])

    # -- training --
    warmup_epochs: int = 3  # phase 1: MMD + soft-DTW only
    epochs: int = 12  # total epochs (phase 2 = epochs - warmup)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    grad_clip: float = 1.0
    patience: int = 5  # early-stop patience (0 disables)
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs cannot exceed epochs")
        if len(self.causal_lambda) != self.causal_heads:
            # Broadcast a single lambda across heads, else require a match.
            if len(self.causal_lambda) == 1:
                self.causal_lambda = self.causal_lambda * self.causal_heads
            else:
                raise ValueError("causal_lambda length must match causal_heads")

    @classmethod
    def quick(cls) -> "SCIDConfig":
        """Small + fast default (seconds to fit). Used by the detector."""
        return cls()

    @classmethod
    def paper(cls) -> "SCIDConfig":
        """Larger, paper-scale capacities and schedule (not a reproduction)."""
        return cls(
            window_size=100,
            step_size=50,
            gru_hidden=64,
            gat_hidden=64,
            gat_heads=4,
            embed_dim=64,
            decoder_hidden=128,
            cmi_top_k=15,
            causal_heads=3,
            causal_lambda=[1.0, 0.5, 0.25],
            warmup_epochs=10,
            epochs=60,
            lr=1e-3,
            batch_size=64,
            patience=10,
        )
