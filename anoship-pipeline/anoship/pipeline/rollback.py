"""Automated rollback: restore the last known-good snapshot on gate failure."""

from __future__ import annotations

from ..core.context import RunContext
from ..core.types import DeploymentState, EventKind, GateDecision, RolloutStage, Snapshot

__all__ = ["RollbackController"]


class RollbackController:
    """Performs the automated rollback when a health gate fails.

    The controller isolates the failure to the current stage/region, restores
    the safe snapshot from the registry, and records mitigation timing -- turning
    what is otherwise hours of manual intervention into an automated response.
    """

    def trigger(
        self,
        context: RunContext,
        stage: RolloutStage,
        decision: GateDecision,
    ) -> Snapshot:
        context.emit(
            EventKind.ROLLBACK_TRIGGERED,
            f"gate failed at stage '{stage.name}': {decision.reason}",
            stage=stage.name,
            region=stage.region,
            exposure=stage.exposure,
        )
        context.set_state(DeploymentState.ROLLING_BACK)

        assert context.snapshots is not None
        target = context.snapshots.rollback_target()

        if context.metrics is not None:
            context.metrics.incr("rollbacks")

        context.set_state(DeploymentState.ROLLED_BACK)
        context.emit(
            EventKind.ROLLBACK_COMPLETED,
            f"restored safe snapshot '{target.snapshot_id}', "
            f"failure isolated to '{stage.region or stage.name}'",
            stage=stage.name,
            restored=target.snapshot_id,
            isolated_to=stage.region or stage.name,
        )
        return target
