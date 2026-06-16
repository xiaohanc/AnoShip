"""Causal graph construction for SCID (computed once, cached at ``fit`` start).

For each ordered pair of variables ``(i, j)`` we score a directed causal
strength

    S_ij = sum_h  lambda_h * | A_orig - A_cf | * normCMI

where, at temporal lag ``h``:

* ``A_orig`` is the normalized association between ``i`` and ``j`` on the real
  data,
* ``A_cf`` is the same association on the *counterfactual* data (each variable
  replaced by its own mean -> constant -> association collapses to ~0), so the
  ``| A_orig - A_cf |`` term measures how much of the association is genuinely
  carried by the variable's temporal structure, and
* ``normCMI`` is the conditional normalized mutual information of ``i`` and ``j``
  given the most-correlated other variables (the conditioning set is capped at
  ``cmi_top_k`` for tractability when ``V`` is large -- a documented
  approximation).

An edge is kept when ``S_ij`` exceeds ``mean(S) + std(S)`` over the off-diagonal
entries.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import SCIDConfig
from .cmi import ksg_mi, normalized_cmi, standardize

__all__ = ["build_causal_graph"]


def _normalized_mi(x: np.ndarray, y: np.ndarray, k: int) -> float:
    return float(1.0 - np.exp(-ksg_mi(x, y, k=k)))


def _conditioning_set(Xs: np.ndarray, i: int, j: int, top_k: int) -> List[int]:
    """Up to ``top_k`` other variables most linearly correlated with ``i``."""
    V = Xs.shape[1]
    others = [v for v in range(V) if v != i and v != j]
    if len(others) <= top_k:
        return others
    xi = Xs[:, i]
    scored = []
    for v in others:
        if np.std(Xs[:, v]) > 0 and np.std(xi) > 0:
            c = abs(float(np.corrcoef(xi, Xs[:, v])[0, 1]))
        else:
            c = 0.0
        scored.append((c, v))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [v for _, v in scored[:top_k]]


def _lagged(Xs: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(past, future)`` views aligned for a directed lag (lag 0 = same)."""
    if lag <= 0:
        return Xs, Xs
    return Xs[:-lag], Xs[lag:]


def build_causal_graph(
    X: np.ndarray, config: SCIDConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the cached causal strength matrix and its boolean adjacency.

    Parameters
    ----------
    X:
        Raw training split of shape ``(T, V)``.

    Returns
    -------
    (S, adjacency):
        ``S`` is the ``(V, V)`` causal-strength matrix (zero diagonal);
        ``adjacency`` is the boolean ``(V, V)`` edge mask.
    """
    Xs = standardize(X)
    T, V = Xs.shape
    S = np.zeros((V, V))
    if V < 2:
        return S, np.zeros((V, V), dtype=bool)

    # Counterfactual data: each variable replaced by its (constant) mean. After
    # standardization the mean is 0, so the counterfactual columns are constant.
    Xcf = np.zeros_like(Xs)

    lambdas = config.causal_lambda
    k = config.cmi_k
    for h in range(config.causal_heads):
        past_o, fut_o = _lagged(Xs, h)
        past_c, fut_c = _lagged(Xcf, h)
        for i in range(V):
            for j in range(V):
                if i == j:
                    continue
                cond_idx = _conditioning_set(Xs, i, j, config.cmi_top_k)
                z = past_o[:, cond_idx] if cond_idx else None
                a_orig = _normalized_mi(past_o[:, i], fut_o[:, j], k)
                a_cf = _normalized_mi(past_c[:, i], fut_c[:, j], k)
                ncmi = normalized_cmi(past_o[:, i], fut_o[:, j], z, k=k)
                S[i, j] += lambdas[h] * abs(a_orig - a_cf) * ncmi

    off = ~np.eye(V, dtype=bool)
    vals = S[off]
    thr = float(vals.mean() + vals.std())
    adjacency = (S > thr) & off
    return S, adjacency
