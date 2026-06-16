"""Pluggable ``scid`` detector: a from-paper SCID model behind anoship's interface.

This adapter wires the from-paper SCID model (:mod:`scid_ad`) into anoship's
``Detector`` contract so it can be selected in a deployment pipeline like any
built-in detector. It standardizes the incoming ``(T, C)`` stream, computes and
caches the KSG-CMI causal graph once at ``fit`` start, trains the SCID model with
its two-phase (warm-up -> full) objective, and scores with the average of the
reconstruction- and prediction-pass errors. Per-channel attribution comes from
the DCRE's per-variable contribution weights.

Requires the ``anoship-scid`` distribution (torch, scipy, scikit-learn, pandas).

Implementation note
-------------------
There is no public reference implementation of SCID and no benchmark datasets are
bundled, so this adapter (and the model it wraps) makes **no claim to reproduce
the paper's reported numbers**. It targets live-stream deployment use and is
validated behaviorally on synthetic anomaly injection. The default configuration
is a quick preset (small dims, few epochs); a paper-scale config is available via
``scid_ad.config.SCIDConfig.paper()``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from anoship.core.errors import NotFittedError
from anoship.core.interfaces import Detector, ThresholdStrategy
from anoship.core.registry import register_detector
from anoship.core.types import AnomalyResult
from anoship.scoring.thresholds import SigmaThreshold
from scid_ad.causal.graph import build_causal_graph
from scid_ad.config import SCIDConfig
from scid_ad.masking import coefficient_of_variation, mask_rate
from scid_ad.model import SCIDModel
from scid_ad.pot import POTThreshold  # noqa: F401  registers the "pot" threshold
from scid_ad.training import SCIDTrainer
from sklearn.preprocessing import StandardScaler

__all__ = ["SCIDDetector"]


@register_detector("scid")
class SCIDDetector(Detector):
    name = "scid"

    def __init__(
        self,
        config: Optional[SCIDConfig] = None,
        device: Optional[str] = None,
        threshold: Optional[ThresholdStrategy] = None,
        min_anomaly_rate: float = 0.05,
    ) -> None:
        self.config = config or SCIDConfig.quick()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold_strategy: ThresholdStrategy = threshold or SigmaThreshold()
        self.min_anomaly_rate = float(min_anomaly_rate)

        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[SCIDModel] = None
        self._adj: Optional[torch.Tensor] = None
        self._feature_num = 0
        self._fitted = False

    # ------------------------------------------------------------------ #
    # input preparation (mirrors the mstdf adapter)
    # ------------------------------------------------------------------ #
    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if fit:
            self._scaler = StandardScaler().fit(X)
        assert self._scaler is not None
        return np.nan_to_num(self._scaler.transform(X))

    def _windows(self, n: int) -> list:
        W, step = self.config.window_size, self.config.step_size
        if n <= W:
            return [0]
        starts = list(range(0, n - W + 1, step))
        if starts[-1] != n - W:
            starts.append(n - W)
        return starts

    def _pad(self, A: np.ndarray) -> np.ndarray:
        W = self.config.window_size
        if A.shape[0] >= W:
            return A
        pad = np.repeat(A[-1:], W - A.shape[0], axis=0)
        return np.vstack([A, pad])

    def _prp_heads(self) -> int:
        return 2 if self.config.decoder_hidden % 2 == 0 else 1

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "SCIDDetector":
        torch.manual_seed(self.config.seed)
        Xs = self._scale(X, fit=True)
        n, c = Xs.shape
        self._feature_num = c

        # KSG-CMI causal graph: computed once and cached (preprocessing).
        _, adjacency = build_causal_graph(X, self.config)
        self._adj = torch.tensor(adjacency, dtype=torch.bool, device=self.device)

        # CV-adaptive masking rate from the raw training split.
        cv = coefficient_of_variation(torch.tensor(np.asarray(X, dtype=float)))
        if cv.ndim == 0:
            cv = cv.reshape(1)
        rate = mask_rate(cv, self.config.mask_eta, self.config.mask_cap)

        starts = self._windows(n)
        windows = np.stack(
            [self._pad(Xs[st : st + self.config.window_size]) for st in starts]
        )
        windows_t = torch.tensor(windows, dtype=torch.float32)

        self._model = SCIDModel(self.config, c, prp_heads=self._prp_heads()).to(
            self.device
        )
        trainer = SCIDTrainer(self._model, self.config, device=self.device)
        trainer.fit(windows_t, rate, adj=self._adj)

        self._fitted = True
        base_scores, _ = self._score_arrays(Xs, return_weights=True)
        self.threshold_strategy.fit(base_scores)
        return self

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _score_arrays(self, Xs: np.ndarray, return_weights: bool):
        assert self._model is not None
        n, c = Xs.shape
        sums = np.zeros((n, c))
        counts = np.zeros((n, 1))
        weight_acc = np.zeros(c)
        weight_n = 0
        self._model.eval()
        starts = self._windows(n)
        with torch.no_grad():
            for st in starts:
                sl = slice(st, st + self.config.window_size)
                d_np = self._pad(Xs[sl])
                d = torch.tensor(d_np[None], dtype=torch.float32, device=self.device)
                out = self._model(d, rate=None, adj=self._adj)
                rec_err = (d - out.recon) ** 2
                pred_err = (d - out.pred) ** 2
                err = (0.5 * rec_err + 0.5 * pred_err)[0].cpu().numpy()
                valid = min(self.config.window_size, n - st)
                sums[st : st + valid] += err[:valid]
                counts[st : st + valid] += 1
                weight_acc += out.var_weight[0].cpu().numpy()
                weight_n += 1
        counts[counts == 0] = 1
        per_channel_t = sums / counts
        row_scores = per_channel_t.mean(axis=1)
        if return_weights:
            weights = weight_acc / max(weight_n, 1)
            return row_scores, weights
        return row_scores, None

    def score(self, window: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("scid detector is not fitted")
        Xs = self._scale(window)
        row_scores, _ = self._score_arrays(Xs, return_weights=False)
        return row_scores

    def is_anomaly(self, window: np.ndarray) -> AnomalyResult:
        if not self._fitted:
            raise NotFittedError("scid detector is not fitted")
        Xs = self._scale(window)
        scores, weights = self._score_arrays(Xs, return_weights=True)
        thr = float(self.threshold_strategy.threshold(scores))
        mask = scores > thr
        rate = float(mask.mean()) if scores.size else 0.0
        total = float(weights.sum()) if weights is not None else 0.0
        if weights is not None and total > 0:
            attribution: Dict[str, float] = {
                f"ch{i}": float(v / total) for i, v in enumerate(weights)
            }
        else:
            attribution = {}
        return AnomalyResult(
            scores=scores,
            score=float(scores.max()) if scores.size else 0.0,
            threshold=thr,
            is_anomaly=rate >= self.min_anomaly_rate,
            anomaly_rate=rate,
            attribution=attribution,
            detector=self.name,
            meta={"model": "SCID (from-paper)"},
        )
