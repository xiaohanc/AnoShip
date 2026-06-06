"""DeploymentPipeline: the self-evaluating rollout orchestrator.

Ties the whole framework together into "a deployment pipeline that is not merely
gradual, but self-evaluating": progressive rollout, anomaly-detection-powered
health gating at each stage, automated rollback on failure, and standardized
observability throughout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..attribution.root_cause import RootCause, attribute
from ..core.context import RunContext
from ..core.errors import NotFittedError
from ..core.interfaces import Detector, Exporter, GatePolicy, RolloutStrategy
from ..core.types import (
    DeploymentState,
    Event,
    EventKind,
    GateAction,
    GateDecision,
    RolloutStage,
)
from ..observability.events import EventBus
from ..observability.metrics import MetricsCollector, RunSummary
from ..snapshots.registry import SnapshotRegistry
from .health_gate import HealthGate
from .promotion import PromotionController
from .rollback import RollbackController
from .state_machine import DeploymentStateMachine

__all__ = ["SignalSource", "StaticSource", "CallableSource", "RunReport", "DeploymentPipeline"]


class SignalSource(ABC):
    """Supplies the observed signal window for a given rollout stage."""

    @abstractmethod
    def observe(self, stage: RolloutStage) -> np.ndarray:
        ...


class StaticSource(SignalSource):
    """Replays a pre-built window per stage (indexed by stage order)."""

    def __init__(self, windows: Sequence[np.ndarray]) -> None:
        self.windows = list(windows)

    def observe(self, stage: RolloutStage) -> np.ndarray:
        idx = min(stage.index, len(self.windows) - 1)
        return self.windows[idx]


class CallableSource(SignalSource):
    """Wraps a ``stage -> window`` callable as a signal source."""

    def __init__(self, fn: Callable[[RolloutStage], np.ndarray]) -> None:
        self.fn = fn

    def observe(self, stage: RolloutStage) -> np.ndarray:
        return self.fn(stage)


@dataclass
class RunReport:
    """Everything produced by a single deployment run."""

    summary: RunSummary
    decisions: List[GateDecision]
    events: List[Event]
    metrics: dict
    success: bool
    held: bool
    root_cause: Optional[RootCause] = None
    context: Optional[RunContext] = field(default=None, repr=False)

    @property
    def rolled_back(self) -> bool:
        return self.summary.rolled_back


class DeploymentPipeline:
    def __init__(
        self,
        detector: Detector,
        rollout: RolloutStrategy,
        policy: GatePolicy,
        *,
        exporters: Optional[Sequence[Exporter]] = None,
        risk_tier: str = "standard",
    ) -> None:
        self.detector = detector
        self.rollout = rollout
        self.policy = policy
        self.exporters = list(exporters or [])
        self.risk_tier = risk_tier
        self.gate = HealthGate(detector, policy)
        self.rollback = RollbackController()
        self.promotion = PromotionController()
        self._fitted = False

    def fit(self, baseline: np.ndarray) -> "DeploymentPipeline":
        """Calibrate the detector on anomaly-free baseline signals."""
        self.detector.fit(baseline)
        self._fitted = True
        return self

    def run(
        self,
        source: SignalSource,
        *,
        candidate_id: str = "candidate",
        safe_id: str = "baseline-safe",
        risk_tier: Optional[str] = None,
    ) -> RunReport:
        if not self._fitted:
            raise NotFittedError(
                "pipeline detector is not fitted; call fit(baseline) first"
            )
        bus = EventBus(self.exporters)
        ctx = RunContext(
            risk_tier=risk_tier or self.risk_tier,
            event_bus=bus,
            metrics=MetricsCollector(),
            snapshots=SnapshotRegistry(),
        )
        ctx.snapshots.register(safe_id, safe=True, meta={"role": "baseline"})
        ctx.snapshots.register(candidate_id, meta={"role": "candidate"})
        ctx.metrics.set("stages_promoted", 0)

        sm = DeploymentStateMachine(ctx)
        ctx.emit(
            EventKind.DEPLOYMENT_STARTED,
            f"deploying '{candidate_id}' (risk tier: {ctx.risk_tier})",
            candidate=candidate_id,
            safe=safe_id,
        )
        sm.transition(DeploymentState.INITIALIZING)

        stages = self.rollout.stages()
        decisions: List[GateDecision] = []
        promoted = 0
        rolled_back = False
        detection_stage: Optional[int] = None
        rollback_decision: Optional[GateDecision] = None

        for i, stage in enumerate(stages):
            sm.transition(
                DeploymentState.CANARY if i == 0 else DeploymentState.RAMPING
            )
            ctx.emit(
                EventKind.STAGE_ENTERED,
                f"entering stage '{stage.name}' (exposure {stage.exposure:.0%})",
                stage=stage.name,
                exposure=stage.exposure,
                region=stage.region,
            )
            window = np.asarray(source.observe(stage), dtype=float)
            ctx.emit(
                EventKind.SIGNAL_OBSERVED,
                f"observed {window.shape[0]} samples",
                stage=stage.name,
                samples=int(window.shape[0]),
            )
            decision = self.gate.evaluate(window, stage, ctx)
            decisions.append(decision)

            if decision.action == GateAction.ROLLBACK:
                rolled_back = True
                detection_stage = i
                rollback_decision = decision
                self.rollback.trigger(ctx, stage, decision)
                break
            if decision.action == GateAction.HOLD:
                break
            self.promotion.promote(ctx, stage)
            promoted += 1

        completed = (not rolled_back) and (promoted == len(stages))
        held = (not rolled_back) and (not completed)

        if completed:
            ctx.snapshots.mark_safe(candidate_id)
            sm.transition(DeploymentState.COMPLETED)
            ctx.emit(
                EventKind.DEPLOYMENT_COMPLETED,
                f"'{candidate_id}' fully rolled out and marked safe",
                candidate=candidate_id,
            )

        root_cause: Optional[RootCause] = None
        root_str: Optional[str] = None
        if rollback_decision is not None and rollback_decision.anomaly_result is not None:
            root_cause = attribute(rollback_decision.anomaly_result)
            root_str = root_cause.summary()

        summary = RunSummary(
            deployment_id=ctx.deployment_id,
            final_state=ctx.state.value,
            stages_total=len(stages),
            stages_promoted=promoted,
            rolled_back=rolled_back,
            detection_delay=float(detection_stage)
            if detection_stage is not None
            else None,
            mitigation_time=1.0 if rolled_back else None,
            root_cause=root_str,
            extra={"held": held},
        )
        bus.flush()
        return RunReport(
            summary=summary,
            decisions=decisions,
            events=list(bus.history),
            metrics=ctx.metrics.snapshot(),
            success=completed,
            held=held,
            root_cause=root_cause,
            context=ctx,
        )
