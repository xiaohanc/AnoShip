"""scid_ad -- a from-paper PyTorch implementation of SCID.

SCID (Spatiotemporal Causal Inference Detector) is described in Chen, Xiao,
Zeng, Zhang, Xiao, "SCID: A Spatiotemporal Causal Inference Detector for
multivariate time series anomaly detection", Knowledge-Based Systems 2025.

There is no public reference implementation; this package is an independent
interpretation of the paper. See the distribution ``NOTICE`` for attribution and
the (absence of any) reproduction claim.
"""

from __future__ import annotations

from .config import SCIDConfig

__all__ = ["SCIDConfig"]
