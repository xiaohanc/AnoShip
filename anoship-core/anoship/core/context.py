"""RunContext: shared state threaded through a single deployment run.

The orchestration layer creates one ``RunContext`` per deployment and passes it
to gates, policies, and controllers so they can emit events, record metrics, and
inspect deployment state without global variables.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from .types import DeploymentState, Event, EventKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..observability.events import EventBus
    from ..observability.metrics import MetricsCollector
    from ..snapshots.registry import SnapshotRegistry

__all__ = ["RunContext"]


@dataclass
class RunContext:
    """Carries cross-cutting state for one deployment run."""

    deployment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: DeploymentState = DeploymentState.PENDING
    risk_tier: str = "standard"
    event_bus: Optional["EventBus"] = None
    metrics: Optional["MetricsCollector"] = None
    snapshots: Optional["SnapshotRegistry"] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def emit(
        self,
        kind: EventKind,
        message: str = "",
        stage: Optional[str] = None,
        **payload: Any,
    ) -> Event:
        """Create an event and publish it on the bus (if attached)."""
        event = Event(kind=kind, message=message, stage=stage, payload=payload)
        if self.event_bus is not None:
            self.event_bus.publish(event)
        return event

    def set_state(self, state: DeploymentState) -> None:
        """Transition deployment state and emit a STATE_CHANGED event."""
        previous = self.state
        self.state = state
        self.emit(
            EventKind.STATE_CHANGED,
            f"{previous.value} -> {state.value}",
            **{"from": previous.value, "to": state.value},
        )
