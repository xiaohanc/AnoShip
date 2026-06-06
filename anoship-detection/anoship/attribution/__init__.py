"""Attribution layer: channel-level root-cause analysis of anomalies."""

from __future__ import annotations

from .root_cause import aggregate_attribution, attribute, RootCause

__all__ = ["RootCause", "aggregate_attribution", "attribute"]
