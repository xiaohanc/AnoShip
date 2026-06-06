"""Exception hierarchy for anoship.

A single, well-defined exception tree makes failures explicit and lets callers
distinguish configuration mistakes from genuine runtime safety events.
"""

from __future__ import annotations

__all__ = [
    "AnoshipError",
    "ConfigError",
    "RegistryError",
    "DetectorError",
    "NotFittedError",
    "PipelineError",
    "RollbackError",
    "PolicyError",
]


class AnoshipError(Exception):
    """Base class for all anoship errors."""


class ConfigError(AnoshipError):
    """Raised when a configuration is invalid or incomplete."""


class RegistryError(AnoshipError):
    """Raised when a plugin name cannot be resolved or is double-registered."""


class DetectorError(AnoshipError):
    """Raised for detector-level failures."""


class NotFittedError(DetectorError):
    """Raised when a detector is scored before being fitted."""


class PipelineError(AnoshipError):
    """Raised for orchestration-level failures."""


class RollbackError(PipelineError):
    """Raised when a rollback cannot be completed safely."""


class PolicyError(AnoshipError):
    """Raised when a gate policy is misconfigured."""
