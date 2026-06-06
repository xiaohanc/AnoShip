"""Root-cause attribution over anomaly results.

Detection answers "is something wrong?"; attribution answers "where?". This
turns the per-channel contributions a detector emits into a ranked, readable
root-cause hint, and can aggregate attribution across an entire deployment run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..core.types import AnomalyResult

__all__ = ["RootCause", "attribute", "aggregate_attribution"]


@dataclass
class RootCause:
    ranking: List[Tuple[str, float]]

    @property
    def top_channel(self) -> str:
        return self.ranking[0][0] if self.ranking else "unknown"

    def summary(self, top_k: int = 3) -> str:
        if not self.ranking:
            return "no attribution available"
        parts = [f"{name} ({weight:.0%})" for name, weight in self.ranking[:top_k]]
        return "likely source: " + ", ".join(parts)


def attribute(result: AnomalyResult, top_k: int = 3) -> RootCause:
    """Rank channels by their contribution to ``result``."""
    attribution = result.attribution or {}
    ranking = sorted(attribution.items(), key=lambda kv: kv[1], reverse=True)
    return RootCause(ranking=ranking[: top_k if top_k > 0 else len(ranking)])


def aggregate_attribution(
    results: Sequence[AnomalyResult],
) -> Dict[str, float]:
    """Average channel attributions across many anomaly results."""
    totals: Dict[str, float] = {}
    n = 0
    for res in results:
        if not res.attribution:
            continue
        n += 1
        for name, weight in res.attribution.items():
            totals[name] = totals.get(name, 0.0) + weight
    if n == 0:
        return {}
    grand = sum(totals.values())
    if grand <= 0:
        return totals
    return {name: weight / grand for name, weight in totals.items()}
