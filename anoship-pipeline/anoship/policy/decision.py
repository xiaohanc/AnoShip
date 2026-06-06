"""Core gate-decision logic shared by all gate policies.

Encapsulates the rule that turns an :class:`AnomalyResult` into a promote / hold
/ rollback action. The key idea beyond a raw threshold is *persistence*: a
sustained run of anomalous points is treated very differently from intermittent
spikes, which is what makes the gate robust to noisy telemetry.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.types import AnomalyResult, GateAction

__all__ = ["longest_run", "decide"]


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive ``True`` values in ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def decide(
    result: AnomalyResult,
    *,
    max_anomaly_rate: float,
    min_persistence: int,
    severity_ratio: float,
    hold_fraction: float = 0.5,
) -> Tuple[GateAction, str]:
    """Return the gate action and a human-readable reason for ``result``."""
    mask = result.scores > result.threshold
    run = longest_run(mask)
    rate = result.anomaly_rate
    peak_ratio = (
        result.score / result.threshold if result.threshold > 0 else float("inf")
    )

    sustained = run >= min_persistence
    rate_breach = rate >= max_anomaly_rate
    severe = peak_ratio >= severity_ratio

    if sustained and (rate_breach or severe):
        return (
            GateAction.ROLLBACK,
            f"sustained anomaly (run={run}, rate={rate:.0%}, "
            f"peak={peak_ratio:.1f}x threshold)",
        )
    if rate >= max_anomaly_rate * hold_fraction:
        return (
            GateAction.HOLD,
            f"elevated but unconfirmed (run={run}, rate={rate:.0%})",
        )
    return (
        GateAction.PROMOTE,
        f"healthy (run={run}, rate={rate:.0%}, peak={peak_ratio:.1f}x)",
    )
