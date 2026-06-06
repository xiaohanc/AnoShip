"""Risk tiers: stricter gating for higher-impact deployments.

Inspired by risk-based AI governance guidance (e.g. OMB M-25-21's "minimum risk
management practices" for higher-impact AI and the NIST AI RMF), each tier maps
an impact level to concrete gate thresholds. A higher-impact deployment tolerates
fewer anomalies, requires less persistence before acting, and reacts to smaller
severity excursions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..core.errors import PolicyError

__all__ = ["RiskTier", "TIERS", "get_tier"]


@dataclass(frozen=True)
class RiskTier:
    name: str
    #: Max fraction of anomalous points tolerated before action.
    max_anomaly_rate: float
    #: Consecutive anomalous points required to act (debounces noise).
    min_persistence: int
    #: Peak score / threshold ratio that counts as a severe excursion.
    severity_ratio: float
    #: Fraction of ``max_anomaly_rate`` above which the gate holds (slows down).
    hold_fraction: float = 0.5


TIERS: Dict[str, RiskTier] = {
    # Lenient: experimentation / low-impact surfaces.
    "experimental": RiskTier("experimental", 0.25, 5, 3.5, 0.6),
    # Default operating point.
    "standard": RiskTier("standard", 0.12, 3, 2.5, 0.5),
    # Strict: higher-impact AI, per risk-based governance guidance.
    "high_impact": RiskTier("high_impact", 0.05, 2, 1.8, 0.4),
}


def get_tier(name: Optional[str]) -> RiskTier:
    key = (name or "standard").lower()
    if key not in TIERS:
        raise PolicyError(f"unknown risk tier {name!r}; available: {', '.join(TIERS)}")
    return TIERS[key]
