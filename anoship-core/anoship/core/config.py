"""Declarative configuration schema for anoship deployments.

A deployment can be described entirely in YAML (or a plain dict), which the
registry then resolves into concrete components. This is what lets an
organization adopt a vetted deployment-safety policy without writing code -- a
core goal of the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import ConfigError

__all__ = ["ComponentSpec", "DeploymentConfig"]


@dataclass
class ComponentSpec:
    """A reference to a registered component plus its constructor params."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, value: Any, what: str) -> "ComponentSpec":
        if value is None:
            raise ConfigError(f"missing required component: {what}")
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, dict):
            if "name" not in value:
                raise ConfigError(f"{what} spec must include a 'name' field")
            params = dict(value)
            name = params.pop("name")
            # Allow an explicit nested 'params' block as well.
            nested = params.pop("params", {})
            params.update(nested or {})
            return cls(name=name, params=params)
        raise ConfigError(f"invalid {what} spec: {value!r}")


@dataclass
class DeploymentConfig:
    """Top-level specification of a deployment-safety run."""

    detector: ComponentSpec
    rollout: ComponentSpec
    policy: ComponentSpec
    threshold: Optional[ComponentSpec] = None
    scenario: Optional[ComponentSpec] = None
    exporters: List[ComponentSpec] = field(default_factory=list)
    risk_tier: str = "standard"
    seed: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentConfig":
        if not isinstance(data, dict):
            raise ConfigError("deployment config must be a mapping")
        exporters_raw = data.get("exporters", []) or []
        if isinstance(exporters_raw, (str, dict)):
            exporters_raw = [exporters_raw]
        return cls(
            detector=ComponentSpec.parse(data.get("detector"), "detector"),
            rollout=ComponentSpec.parse(data.get("rollout"), "rollout"),
            policy=ComponentSpec.parse(data.get("policy"), "policy"),
            threshold=(
                ComponentSpec.parse(data["threshold"], "threshold")
                if data.get("threshold") is not None
                else None
            ),
            scenario=(
                ComponentSpec.parse(data["scenario"], "scenario")
                if data.get("scenario") is not None
                else None
            ),
            exporters=[ComponentSpec.parse(e, "exporter") for e in exporters_raw],
            risk_tier=data.get("risk_tier", "standard"),
            seed=int(data.get("seed", 0)),
            meta=data.get("meta", {}) or {},
        )

    @classmethod
    def from_yaml(cls, path: str) -> "DeploymentConfig":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ConfigError(
                "PyYAML is required to load YAML configs; install with "
                "'pip install anoship[config]'"
            ) from exc
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_dict(data)
