"""Gate policies: pluggable strategies for ruling on a stage's health."""

from __future__ import annotations

from typing import Optional

from ..core.interfaces import GatePolicy
from ..core.registry import register_policy
from ..core.types import AnomalyResult, GateDecision, RolloutStage
from .decision import decide
from .risk_tier import get_tier

__all__ = ["ThresholdGatePolicy", "RiskAwareGatePolicy"]


@register_policy("threshold")
class ThresholdGatePolicy(GatePolicy):
    """A directly-parameterized gate policy with explicit thresholds."""

    name = "threshold"

    def __init__(
        self,
        max_anomaly_rate: float = 0.12,
        min_persistence: int = 3,
        severity_ratio: float = 2.5,
        hold_fraction: float = 0.5,
    ) -> None:
        self.max_anomaly_rate = float(max_anomaly_rate)
        self.min_persistence = int(min_persistence)
        self.severity_ratio = float(severity_ratio)
        self.hold_fraction = float(hold_fraction)

    def evaluate(
        self, anomaly_result: AnomalyResult, stage: RolloutStage, context
    ) -> GateDecision:
        action, reason = decide(
            anomaly_result,
            max_anomaly_rate=self.max_anomaly_rate,
            min_persistence=self.min_persistence,
            severity_ratio=self.severity_ratio,
            hold_fraction=self.hold_fraction,
        )
        return GateDecision(
            action=action,
            reason=reason,
            stage=stage,
            anomaly_result=anomaly_result,
            policy=self.name,
        )


@register_policy("risk_aware")
class RiskAwareGatePolicy(GatePolicy):
    """Derives gate thresholds from the deployment's risk tier.

    The tier comes from the policy's own ``tier`` argument if given, otherwise
    from ``context.risk_tier`` -- so the same policy object adapts its strictness
    to how high-impact the deployment is.
    """

    name = "risk_aware"

    def __init__(self, tier: Optional[str] = None) -> None:
        self.tier_name = tier

    def evaluate(
        self, anomaly_result: AnomalyResult, stage: RolloutStage, context
    ) -> GateDecision:
        tier = get_tier(self.tier_name or getattr(context, "risk_tier", "standard"))
        action, reason = decide(
            anomaly_result,
            max_anomaly_rate=tier.max_anomaly_rate,
            min_persistence=tier.min_persistence,
            severity_ratio=tier.severity_ratio,
            hold_fraction=tier.hold_fraction,
        )
        return GateDecision(
            action=action,
            reason=f"[tier={tier.name}] {reason}",
            stage=stage,
            anomaly_result=anomaly_result,
            policy=self.name,
        )
