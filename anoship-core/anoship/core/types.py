"""Core data types shared across every anoship subsystem.

These are intentionally dependency-free (NumPy only) value objects so that the
detection layer, the orchestration layer, the policy layer, and the
observability layer can all exchange information through a single, stable
vocabulary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "GateAction",
    "DeploymentState",
    "EventKind",
    "AnomalyResult",
    "RolloutStage",
    "GateDecision",
    "Snapshot",
    "Event",
]


class GateAction(str, Enum):
    """The decision a health gate can emit for a rollout stage."""

    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


class DeploymentState(str, Enum):
    """States of the deployment state machine."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    CANARY = "canary"
    RAMPING = "ramping"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class EventKind(str, Enum):
    """Categories of structured observability events."""

    DEPLOYMENT_STARTED = "deployment_started"
    STAGE_ENTERED = "stage_entered"
    SIGNAL_OBSERVED = "signal_observed"
    GATE_EVALUATED = "gate_evaluated"
    PROMOTED = "promoted"
    ROLLBACK_TRIGGERED = "rollback_triggered"
    ROLLBACK_COMPLETED = "rollback_completed"
    DEPLOYMENT_COMPLETED = "deployment_completed"
    STATE_CHANGED = "state_changed"


@dataclass
class AnomalyResult:
    """Output of a :class:`~anoship.core.interfaces.Detector` over a window.

    Attributes
    ----------
    scores:
        Per-timestep (or per-sample) anomaly scores for the evaluated window.
    score:
        A single aggregate score for the window (higher == more anomalous).
    threshold:
        The decision threshold applied to ``scores``.
    is_anomaly:
        Whether the window is considered anomalous overall.
    anomaly_rate:
        Fraction of points in the window that exceeded ``threshold``.
    attribution:
        Optional per-channel contribution to the anomaly (root-cause hint).
    detector:
        Name of the detector that produced this result.
    meta:
        Free-form detector-specific diagnostics.
    """

    scores: np.ndarray
    score: float
    threshold: float
    is_anomaly: bool
    anomaly_rate: float = 0.0
    attribution: Optional[Dict[str, float]] = None
    detector: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """Signed distance of the aggregate score from the threshold."""
        return float(self.score - self.threshold)


@dataclass
class RolloutStage:
    """A single stage of a progressive rollout."""

    name: str
    exposure: float  # fraction of traffic exposed, in [0, 1]
    index: int = 0
    region: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.exposure <= 1.0:
            raise ValueError(f"exposure must be within [0, 1], got {self.exposure!r}")


@dataclass
class GateDecision:
    """A health-gate ruling for a particular rollout stage."""

    action: GateAction
    reason: str
    stage: RolloutStage
    anomaly_result: Optional[AnomalyResult] = None
    policy: str = "default"
    timestamp: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return self.action == GateAction.PROMOTE


@dataclass
class Snapshot:
    """A versioned model snapshot tracked by the deployment."""

    snapshot_id: str
    version: int = 0
    is_safe: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """A structured observability event on the deployment timeline."""

    kind: EventKind
    message: str = ""
    stage: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "stage": self.stage,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }
