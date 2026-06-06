"""Pluggable `mstdf` detector: the real MSTDF-AD model behind anoship's interface.

This adapter wires the vendored, published MSTDF-AD model
(:mod:`mstdf_ad`) into anoship's ``Detector`` contract so it can be selected in a
deployment pipeline like any built-in detector. It standardizes the incoming
``(T, C)`` stream, builds the calendar ``time`` features and the wavelet
seasonal/trend (``S``/``T``) components the model expects, trains with the
upstream KL + reconstruction objective, and scores with the upstream
reconstruction-error x KL-contribution formula.

Requires the ``anoship-mstdf`` distribution (torch, PyWavelets, ...).

Reproducibility note
--------------------
For the *exact* published numbers, run the vendored pipeline directly
(``python -m mstdf_ad.main --dataset PSM``). This adapter targets live-stream
deployment use and uses a simple linear KL (beta) ramp rather than the upstream
adaptive beta schedule; that is the only intentional deviation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from anoship.core.errors import NotFittedError
from anoship.core.interfaces import Detector, ThresholdStrategy
from anoship.core.registry import register_detector
from anoship.core.types import AnomalyResult
from anoship.scoring.thresholds import SigmaThreshold

from mstdf_ad.data.preprocess import getTimeEmbedding
from mstdf_ad.model.MSTDF import MSTDF
from mstdf_ad.utils.dwt import dwt
from mstdf_ad.utils.getKLContributions import kl_divergence_contributions

__all__ = ["MSTDFDetector"]


def _synthetic_time(n: int) -> np.ndarray:
    """Build the 6 calendar time-features the model expects, for n steps."""
    ts = pd.date_range("2024-01-01", periods=n, freq="s")
    return getTimeEmbedding(ts.values)  # (n, 6)


@register_detector("mstdf")
class MSTDFDetector(Detector):
    name = "mstdf"

    def __init__(
        self,
        window_size: int = 64,
        step_size: int = 16,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,
        model_dim: int = 512,
        ff_dim: int = 2048,
        atten_dim: int = 64,
        hidden_dim: int = 5,
        kernel_size: int = 16,
        stride: int = 8,
        block_num: int = 2,
        head_num: int = 8,
        dropout: float = 0.4,
        epsilon: float = 0.2,
        p: float = 1.0,
        wavelet: str = "db4",
        level: int = 5,
        window_length: int = 51,
        polyorder: int = 3,
        seed: int = 1234,
        device: Optional[str] = None,
        threshold: Optional[ThresholdStrategy] = None,
        min_anomaly_rate: float = 0.05,
    ) -> None:
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.model_kwargs = dict(
            model_dim=model_dim, ff_dim=ff_dim, atten_dim=atten_dim,
            hidden_dim=hidden_dim, kernel_size=kernel_size, stride=stride,
            block_num=block_num, head_num=head_num, dropout=dropout,
        )
        self.epsilon = float(epsilon)
        self.p = float(p)
        self.wavelet = wavelet
        self.level = int(level)
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.seed = int(seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold_strategy: ThresholdStrategy = threshold or SigmaThreshold()
        self.min_anomaly_rate = float(min_anomaly_rate)

        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[nn.Module] = None
        self._p_tensor: Optional[torch.Tensor] = None
        self._feature_num = 0
        self._fitted = False

    # ------------------------------------------------------------------ #
    # input preparation
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
        """Start indices of fixed-size windows covering [0, n)."""
        W, step = self.window_size, self.step_size
        if n <= W:
            return [0]
        starts = list(range(0, n - W + 1, step))
        if starts[-1] != n - W:
            starts.append(n - W)
        return starts

    def _pad(self, A: np.ndarray) -> np.ndarray:
        """Edge-pad a short array up to window_size rows."""
        if A.shape[0] >= self.window_size:
            return A
        pad = np.repeat(A[-1:], self.window_size - A.shape[0], axis=0)
        return np.vstack([A, pad])

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "MSTDFDetector":
        torch.manual_seed(self.seed)
        Xs = self._scale(X, fit=True)
        n, c = Xs.shape
        self._feature_num = c

        var = np.var(Xs, axis=0)
        init_p = np.maximum(self.p * var, 1.0)
        self._p_tensor = torch.tensor(init_p, dtype=torch.float32, device=self.device)

        trend, season = dwt(
            Xs, self.wavelet, self.level, True, self.window_length, self.polyorder
        )
        time = _synthetic_time(n)

        starts = self._windows(n)
        data_w, time_w, s_w, t_w = [], [], [], []
        for st in starts:
            sl = slice(st, st + self.window_size)
            data_w.append(self._pad(Xs[sl]))
            time_w.append(self._pad(time[sl]))
            s_w.append(self._pad(season[sl]))
            t_w.append(self._pad(trend[sl]))
        data_w = torch.tensor(np.stack(data_w), dtype=torch.float32)
        time_w = torch.tensor(np.stack(time_w), dtype=torch.float32)
        s_w = torch.tensor(np.stack(s_w), dtype=torch.float32)
        t_w = torch.tensor(np.stack(t_w), dtype=torch.float32)

        self._model = MSTDF(
            dataset="custom",
            window_size=self.window_size,
            feature_num=c,
            time_num=time.shape[1],
            device=self.device,
            epsilon=self.epsilon,
            **self.model_kwargs,
        ).to(self.device)
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        mse = nn.MSELoss(reduction="mean")
        free_bits = self.model_kwargs["hidden_dim"] / 10.0

        num = data_w.shape[0]
        self._model.train()
        for epoch in range(self.epochs):
            beta = (epoch + 1) / self.epochs  # linear KL ramp (see module docstring)
            perm = torch.randperm(num)
            for i in range(0, num, self.batch_size):
                idx = perm[i : i + self.batch_size]
                d = data_w[idx].to(self.device)
                tm = time_w[idx].to(self.device)
                bs = s_w[idx].to(self.device)
                bt = t_w[idx].to(self.device)

                optimizer.zero_grad()
                mu_x, mu_y, lv_x, lv_y, _, stable, trend_out, recon = self._model(
                    d, tm, self._p_tensor
                )
                mu = mu_x.reshape(-1, self.model_kwargs["hidden_dim"])
                lv = lv_x.reshape(-1, self.model_kwargs["hidden_dim"])
                kl = torch.mean(torch.sum(-0.5 * (1 + lv - torch.exp(lv) - mu**2), 1))

                recon_f = torch.fft.fftn(recon, dim=(1, 2))
                data_f = torch.fft.fftn(d, dim=(1, 2))
                mse_real = torch.mean((recon_f.real - data_f.real) ** 2)
                mse_imag = torch.mean((recon_f.imag - data_f.imag) ** 2)

                recon_loss = (
                    mse(recon, d) + mse(stable, bs) + mse(trend_out, bt)
                    + mse(mu_x, mu_y) + mse(lv_x, lv_y) + mse_real + mse_imag
                )
                loss = beta * torch.clamp(kl, min=free_bits) + recon_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

        self._fitted = True
        base_scores, _ = self._score_arrays(Xs, time, return_channels=True)
        self.threshold_strategy.fit(base_scores)
        return self

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _score_arrays(self, Xs: np.ndarray, time: np.ndarray, return_channels: bool):
        assert self._model is not None
        n, c = Xs.shape
        sums = np.zeros((n, c))
        counts = np.zeros((n, 1))
        self._model.eval()
        starts = self._windows(n)
        with torch.no_grad():
            for st in starts:
                sl = slice(st, st + self.window_size)
                d_np = self._pad(Xs[sl])
                tm_np = self._pad(time[sl])
                d = torch.tensor(d_np[None], dtype=torch.float32, device=self.device)
                tm = torch.tensor(tm_np[None], dtype=torch.float32, device=self.device)
                *_, recon = self._model(d, tm, 0)
                contrib = kl_divergence_contributions(recon, d)
                err = ((d - recon) ** 2 * torch.abs(contrib))[0].cpu().numpy()
                valid = min(self.window_size, n - st)
                sums[st : st + valid] += err[:valid]
                counts[st : st + valid] += 1
        counts[counts == 0] = 1
        per_channel_t = sums / counts
        row_scores = per_channel_t.mean(axis=1)
        channel_mean = per_channel_t.mean(axis=0)
        if return_channels:
            return row_scores, channel_mean
        return row_scores, None

    def score(self, window: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("mstdf detector is not fitted")
        Xs = self._scale(window)
        time = _synthetic_time(Xs.shape[0])
        row_scores, _ = self._score_arrays(Xs, time, return_channels=False)
        return row_scores

    def is_anomaly(self, window: np.ndarray) -> AnomalyResult:
        if not self._fitted:
            raise NotFittedError("mstdf detector is not fitted")
        Xs = self._scale(window)
        time = _synthetic_time(Xs.shape[0])
        scores, channel_mean = self._score_arrays(Xs, time, return_channels=True)
        thr = float(self.threshold_strategy.threshold(scores))
        mask = scores > thr
        rate = float(mask.mean()) if scores.size else 0.0
        total = float(channel_mean.sum()) if channel_mean is not None else 0.0
        if channel_mean is not None and total > 0:
            attribution: Dict[str, float] = {
                f"ch{i}": float(v / total) for i, v in enumerate(channel_mean)
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
            meta={"model": "MSTDF-AD"},
        )
