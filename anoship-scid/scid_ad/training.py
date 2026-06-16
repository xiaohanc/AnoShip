"""Two-phase trainer for SCID.

The model is trained in two phases (Cuturi-style curriculum):

* **warm-up** -- only the soft-DTW (granularity) and MMD (counterfactual
  differentiation) terms are active, so the encoder/causal representation is
  shaped before the decoder is asked to reconstruct against it, and
* **full** -- the Predictive Reconstruction Process term is added and the full
  ``L_total = alpha*L_PRP + beta*L_DTW + gamma*L_MMD`` is optimized.

Early stopping watches the full-phase loss and restores the best weights.

MMD sample sets
---------------
MMD is a two-sample statistic, so the *pull* term needs two distinct samples of
the factual distribution. We split each mini-batch in half: one half is the
query set ``z``, the other the positive (factual) reference ``z_pos``; the
counterfactual latents form the negative set ``z_neg``. With fewer than four
windows in a batch the pull term is skipped (``z_pos = z``), leaving only the
push.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
from torch import optim

from .config import SCIDConfig
from .losses import SCIDLoss
from .model import SCIDModel

__all__ = ["SCIDTrainer"]


class SCIDTrainer:
    def __init__(
        self,
        model: SCIDModel,
        config: SCIDConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.loss_fn = SCIDLoss(config).to(self.device)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def _mmd_sets(self, z_fact: torch.Tensor, z_cf: torch.Tensor):
        B = z_fact.shape[0]
        if B >= 4:
            half = B // 2
            return z_fact[:half], z_fact[half : 2 * half], z_cf[:half]
        return z_fact, z_fact, z_cf

    def fit(
        self,
        windows: torch.Tensor,
        rate: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, float]]:
        """Train on ``windows`` of shape ``(N, L, V)``; return per-epoch history."""
        cfg = self.config
        windows = windows.to(self.device)
        rate = rate.to(self.device)
        if adj is not None:
            adj = adj.to(self.device)
        n = windows.shape[0]

        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        history: List[Dict[str, float]] = []
        best_loss = float("inf")
        best_state = None
        patience_ctr = 0

        for epoch in range(cfg.epochs):
            warmup = epoch < cfg.warmup_epochs
            self.model.train()
            perm = torch.randperm(n, generator=gen)
            agg = {"prp": 0.0, "dtw": 0.0, "mmd": 0.0, "total": 0.0}
            n_batches = 0
            for i in range(0, n, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                batch_x = windows[idx]
                out = self.model(batch_x, rate=rate, adj=adj, generator=gen)
                z, z_pos, z_neg = self._mmd_sets(out.z_fact, out.z_cf)
                total, parts = self.loss_fn(
                    recon=out.recon,
                    pred=out.pred,
                    target=batch_x,
                    z=z,
                    z_pos=z_pos,
                    z_neg=z_neg,
                    warmup=warmup,
                )
                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=cfg.grad_clip
                )
                self.optimizer.step()
                for key in agg:
                    agg[key] += parts[key]
                n_batches += 1

            for key in agg:
                agg[key] /= max(n_batches, 1)
            record = {"epoch": float(epoch), "warmup": float(warmup), **agg}
            history.append(record)

            if not warmup:
                if agg["total"] < best_loss - 1e-5:
                    best_loss = agg["total"]
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if cfg.patience > 0 and patience_ctr >= cfg.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history
