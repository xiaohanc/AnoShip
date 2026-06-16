"""KSG k-nearest-neighbour (conditional) mutual information.

SCID builds its causal graph from conditional mutual information (CMI) between
variables. We use the Kraskov-Stogbauer-Grassberger (KSG) k-NN estimator with
the Chebyshev (L-infinity) norm and digamma corrections -- the Frenzel-Pompe
extension for the *conditional* case:

    I(X;Y|Z) = psi(k) + < psi(n_z + 1) - psi(n_xz + 1) - psi(n_yz + 1) >

and, when there is no conditioning set, the original KSG mutual information:

    I(X;Y) = psi(k) + psi(N) - < psi(n_x + 1) + psi(n_y + 1) >

Inputs are standardized on the training split (constant signals are guarded so
they yield ~0 information rather than NaN/inf). A bounded ``normalized_cmi`` in
``[0, 1]`` is provided for use as an edge weight in the causal graph.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

__all__ = ["standardize", "ksg_mi", "ksg_cmi", "normalized_cmi"]


def _as2d(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None:
        return None
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def standardize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score columns; constant columns (std < eps) collapse to zero."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    out = np.zeros_like(X)
    nonconst = std > eps
    out[:, nonconst] = (X[:, nonconst] - mean[nonconst]) / std[nonconst]
    return out


def _add_jitter(a: np.ndarray, scale: float = 1e-10) -> np.ndarray:
    """Tiny deterministic jitter to break ties for the k-NN radius search."""
    rng = np.random.default_rng(0)
    return a + rng.normal(0.0, scale, size=a.shape)


def _count_within(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """For each point, count *other* points with Chebyshev distance < radius."""
    tree = cKDTree(points)
    counts = np.empty(len(points), dtype=float)
    for i, (p, r) in enumerate(zip(points, radii)):
        # query_ball_point with p=inf (Chebyshev); subtract the point itself.
        n = len(tree.query_ball_point(p, r=r - 1e-12, p=np.inf))
        counts[i] = max(n - 1, 0)
    return counts


def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """KSG mutual information ``I(X;Y)`` (>= ~0; clamped at 0)."""
    x, y = _as2d(x), _as2d(y)
    n = x.shape[0]
    if n <= k + 1:
        return 0.0
    joint = _add_jitter(np.hstack([x, y]))
    tree = cKDTree(joint)
    # distance to the k-th neighbour (index k because the 1st is the point itself)
    dists, _ = tree.query(joint, k=k + 1, p=np.inf)
    eps = dists[:, k]
    nx = _count_within(_add_jitter(x), eps)
    ny = _count_within(_add_jitter(y), eps)
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0))


def ksg_cmi(
    x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None, k: int = 5
) -> float:
    """KSG conditional mutual information ``I(X;Y|Z)``.

    With ``z is None`` (or empty) this reduces to :func:`ksg_mi`. Result is
    clamped at 0 (the estimator can be slightly negative for independent data).
    """
    z = _as2d(z)
    if z is None or z.shape[1] == 0:
        return ksg_mi(x, y, k=k)
    x, y = _as2d(x), _as2d(y)
    n = x.shape[0]
    if n <= k + 1:
        return 0.0
    xz = _add_jitter(np.hstack([x, z]))
    yz = _add_jitter(np.hstack([y, z]))
    zz = _add_jitter(z)
    joint = _add_jitter(np.hstack([x, y, z]))
    tree = cKDTree(joint)
    dists, _ = tree.query(joint, k=k + 1, p=np.inf)
    eps = dists[:, k]
    n_xz = _count_within(xz, eps)
    n_yz = _count_within(yz, eps)
    n_z = _count_within(zz, eps)
    cmi = digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))
    return float(max(cmi, 0.0))


def normalized_cmi(
    x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None, k: int = 5
) -> float:
    """Conditional MI squashed into ``[0, 1)`` via ``1 - exp(-CMI)``.

    Monotonic in the (non-negative) CMI: ~0 for independent variables, growing
    toward 1 as dependence strengthens. Convenient as a bounded edge weight.
    """
    cmi = ksg_cmi(x, y, z, k=k)
    return float(1.0 - np.exp(-cmi))
