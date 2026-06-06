"""Deployment state machine with explicit, validated transitions."""

from __future__ import annotations

from typing import Dict, Set

from ..core.context import RunContext
from ..core.errors import PipelineError
from ..core.types import DeploymentState

__all__ = ["DeploymentStateMachine"]

_S = DeploymentState

_TRANSITIONS: Dict[DeploymentState, Set[DeploymentState]] = {
    _S.PENDING: {_S.INITIALIZING, _S.FAILED},
    _S.INITIALIZING: {_S.CANARY, _S.RAMPING, _S.FAILED},
    _S.CANARY: {_S.RAMPING, _S.COMPLETED, _S.ROLLING_BACK, _S.FAILED},
    _S.RAMPING: {_S.RAMPING, _S.COMPLETED, _S.ROLLING_BACK, _S.FAILED},
    _S.ROLLING_BACK: {_S.ROLLED_BACK, _S.FAILED},
    _S.ROLLED_BACK: set(),
    _S.COMPLETED: set(),
    _S.FAILED: set(),
}


class DeploymentStateMachine:
    """Guards deployment-state transitions and applies them to the context."""

    def __init__(self, context: RunContext) -> None:
        self.context = context

    def transition(self, to: DeploymentState) -> None:
        current = self.context.state
        if to not in _TRANSITIONS.get(current, set()):
            raise PipelineError(
                f"illegal transition {current.value} -> {to.value}"
            )
        self.context.set_state(to)
