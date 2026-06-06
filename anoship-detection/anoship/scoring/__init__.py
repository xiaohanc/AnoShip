"""Scoring layer: thresholding, score fusion, and calibration metrics."""

from __future__ import annotations

from .calibration import best_threshold, DetectionMetrics, evaluate
from .fusion import fuse_scores, robust_zscale
from .thresholds import (
    AdaptiveThreshold,
    QuantileThreshold,
    SigmaThreshold,
    StaticThreshold,
)

__all__ = [
    "DetectionMetrics",
    "best_threshold",
    "evaluate",
    "fuse_scores",
    "robust_zscale",
    "AdaptiveThreshold",
    "QuantileThreshold",
    "SigmaThreshold",
    "StaticThreshold",
]
