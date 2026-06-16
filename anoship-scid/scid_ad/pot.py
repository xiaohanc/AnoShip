"""Optional Peaks-Over-Threshold (POT / EVT) thresholding for SCID.

This is the paper-faithful alternative to the framework-native ``SigmaThreshold``.
It fits a Generalized Pareto Distribution (GPD) to the exceedances above a high
initial quantile of the baseline scores, then sets the decision threshold for a
target risk level ``q`` via the standard POT formula

    z_q = t + (sigma / xi) * (((n * q) / N_t) ** (-xi) - 1)

where ``t`` is the initial quantile, ``n`` the number of baseline scores, and
``N_t`` the number of exceedances. It degrades gracefully (falling back to a
high quantile) when there are too few exceedances to fit a GPD.
"""

from __future__ import annotations

import numpy as np
from anoship.core.interfaces import ThresholdStrategy
from anoship.core.registry import register_threshold
from scipy.stats import genpareto

__all__ = ["POTThreshold"]


@register_threshold("pot")
class POTThreshold(ThresholdStrategy):
    def __init__(self, init_quantile: float = 0.92, risk: float = 1e-3) -> None:
        if not 0.0 < init_quantile < 1.0:
            raise ValueError("init_quantile must be in (0, 1)")
        if not 0.0 < risk < 1.0:
            raise ValueError("risk must be in (0, 1)")
        self.init_quantile = float(init_quantile)
        self.risk = float(risk)
        self._threshold = 0.0

    def fit(self, scores: np.ndarray) -> "POTThreshold":
        scores = np.asarray(scores, dtype=float).ravel()
        n = scores.size
        if n == 0:
            self._threshold = 0.0
            return self
        t = float(np.quantile(scores, self.init_quantile))
        exceed = scores[scores > t] - t
        n_t = exceed.size
        if n_t < 10:
            # Too few peaks to fit a GPD reliably: fall back to a high quantile.
            self._threshold = float(np.quantile(scores, max(self.init_quantile, 0.99)))
            return self
        xi, _, sigma = genpareto.fit(exceed, floc=0.0)
        if sigma <= 0:
            self._threshold = float(np.quantile(scores, 0.99))
            return self
        ratio = (n * self.risk) / n_t
        if abs(xi) < 1e-6:
            zq = t - sigma * np.log(max(ratio, 1e-12))
        else:
            zq = t + (sigma / xi) * (ratio ** (-xi) - 1.0)
        self._threshold = float(zq)
        return self

    def threshold(self, scores: np.ndarray) -> float:
        return self._threshold
