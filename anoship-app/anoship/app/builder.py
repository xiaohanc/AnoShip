"""Factory helpers that turn declarative config into live components."""

from __future__ import annotations

from typing import List, Optional, Sequence

from anoship.core.config import ComponentSpec, DeploymentConfig
from anoship.core.interfaces import Exporter
from anoship.core.registry import DETECTORS, EXPORTERS, POLICIES, ROLLOUTS, THRESHOLDS
from anoship.pipeline import DeploymentPipeline

__all__ = ["build_threshold", "build_detector", "build_pipeline"]


def build_threshold(spec: Optional[ComponentSpec]):
    if spec is None:
        return None
    return THRESHOLDS.create(spec.name, **spec.params)


def build_detector(spec: ComponentSpec, threshold=None):
    params = dict(spec.params)
    if threshold is not None and "threshold" not in params:
        params["threshold"] = threshold
    return DETECTORS.create(spec.name, **params)


def build_pipeline(
    config: DeploymentConfig,
    exporters: Optional[Sequence[Exporter]] = None,
) -> DeploymentPipeline:
    """Assemble a :class:`DeploymentPipeline` from a declarative config."""
    threshold = build_threshold(config.threshold)
    detector = build_detector(config.detector, threshold)
    rollout = ROLLOUTS.create(config.rollout.name, **config.rollout.params)
    policy = POLICIES.create(config.policy.name, **config.policy.params)
    if exporters is None:
        exporters = [EXPORTERS.create(e.name, **e.params) for e in config.exporters]
    return DeploymentPipeline(
        detector=detector,
        rollout=rollout,
        policy=policy,
        exporters=list(exporters),
        risk_tier=config.risk_tier,
    )
