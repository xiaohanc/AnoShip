"""Core framework primitives: types, interfaces, registry, config, context."""

from __future__ import annotations

from .config import ComponentSpec, DeploymentConfig
from .context import RunContext
from .errors import (
    AnoshipError,
    ConfigError,
    DetectorError,
    NotFittedError,
    PipelineError,
    PolicyError,
    RegistryError,
    RollbackError,
)
from .interfaces import (
    Detector,
    Exporter,
    GatePolicy,
    RolloutStrategy,
    ThresholdStrategy,
)
from .registry import (
    DETECTORS,
    EXPORTERS,
    POLICIES,
    Registry,
    ROLLOUTS,
    SCENARIOS,
    THRESHOLDS,
)
from .types import (
    AnomalyResult,
    DeploymentState,
    Event,
    EventKind,
    GateAction,
    GateDecision,
    RolloutStage,
    Snapshot,
)

__all__ = [
    "ComponentSpec",
    "DeploymentConfig",
    "RunContext",
    "AnoshipError",
    "ConfigError",
    "DetectorError",
    "NotFittedError",
    "PipelineError",
    "PolicyError",
    "RegistryError",
    "RollbackError",
    "Detector",
    "Exporter",
    "GatePolicy",
    "RolloutStrategy",
    "ThresholdStrategy",
    "Registry",
    "DETECTORS",
    "EXPORTERS",
    "POLICIES",
    "ROLLOUTS",
    "SCENARIOS",
    "THRESHOLDS",
    "AnomalyResult",
    "DeploymentState",
    "Event",
    "EventKind",
    "GateAction",
    "GateDecision",
    "RolloutStage",
    "Snapshot",
]
