"""SCID causal-inference components.

* :mod:`~scid_ad.causal.counterfactual` -- counterfactual window construction.
* :mod:`~scid_ad.causal.cmi`           -- KSG k-NN conditional mutual information.
* :mod:`~scid_ad.causal.graph`         -- causal graph builder from CMI scores.
"""

from __future__ import annotations

from .cmi import ksg_cmi, ksg_mi, normalized_cmi, standardize
from .counterfactual import counterfactual
from .graph import build_causal_graph

__all__ = [
    "counterfactual",
    "ksg_mi",
    "ksg_cmi",
    "normalized_cmi",
    "standardize",
    "build_causal_graph",
]
