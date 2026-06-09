"""Pluggable anomaly detectors.

Each published method is implemented as a self-contained detector that conforms
to :class:`anoship.core.interfaces.Detector`. Importing this package registers
all built-in detectors in the global registry.

Paper -> module map
-------------------
* ``habituation``     -> Anti-Drosophila Habituation Stream Clustering / AHSC (IEEE ISCIPT 2025)
* ``causal``          -> Causal-Inference MTS AD (Knowledge-Based Systems 2025)
* ``diffusion``       -> Diffusion denoising / disentanglement (J. Supercomputing 2026)
                         + Diffusion-step attention consistency (KBS 2026)
* ``spatiotemporal``  -> MSTDF-AD spatiotemporal fusion (Inf. Processing & Mgmt 2026)
* ``ewma``            -> simple baseline (control)
"""

from __future__ import annotations

from .base import BaseDetector
from .causal import CausalInferenceDetector
from .diffusion import DiffusionDenoiseDetector
from .ensemble import EnsembleDetector
from .ewma import EWMAResidualDetector
from .habituation import HabituationClusterDetector
from .spatiotemporal import SpatioTemporalFusionDetector

__all__ = [
    "BaseDetector",
    "CausalInferenceDetector",
    "DiffusionDenoiseDetector",
    "EnsembleDetector",
    "EWMAResidualDetector",
    "HabituationClusterDetector",
    "SpatioTemporalFusionDetector",
]
