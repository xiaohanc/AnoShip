"""Health gate: evaluates one rollout stage's signals into a gate decision.

This is the "explicit health gating powered by anomaly detection" component. It
wires a detector to a policy: the detector scores the stage's signal window, and
the policy turns that score into a promote / hold / rollback decision, emitting
the relevant observability events along the way.
"""

from __future__ import annotations

import numpy as np

from ..core.context import RunContext
from ..core.interfaces import Detector, GatePolicy
from ..core.types import EventKind, GateDecision, RolloutStage

__all__ = ["HealthGate"]


class HealthGate:
    def __init__(self, detector: Detector, policy: GatePolicy) -> None:
        self.detector = detector
        self.policy = policy

    def evaluate(
        self, window: np.ndarray, stage: RolloutStage, context: RunContext
    ) -> GateDecision:
        result = self.detector.is_anomaly(window)
        decision = self.policy.evaluate(result, stage, context)

        if context.metrics is not None:
            context.metrics.observe("anomaly_rate", result.anomaly_rate)
            context.metrics.observe("peak_score", result.score)

        context.emit(
            EventKind.GATE_EVALUATED,
            f"{decision.action.value}: {decision.reason}",
            stage=stage.name,
            action=decision.action.value,
            anomaly_rate=round(result.anomaly_rate, 3),
            score=round(result.score, 3),
            threshold=round(result.threshold, 3),
        )
        return decision
