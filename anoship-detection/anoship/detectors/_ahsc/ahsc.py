"""The assembled AHSC online stream clusterer (AHSC, Sec. III).

This wires the paper's pieces into the six-step online procedure:

1. **Preprocess** each point with anti-habituation: sparse fly projection
   (:class:`FlyProjection`) -> winner-take-all -> anti-habituation enhancement
   (:class:`AntiHabituation`).
2. **Initialise** the model from the first data block with k-means++.
3. **Find** the best microcluster for a new point via macrocluster-first search.
4. **Update** that microcluster (core/shell counts, center, radius, weight).
5. **Remove** abnormal microclusters whose weight has decayed.
6. **Build** macroclusters from the cross-connected microclusters.

A point is anomalous to the degree it lands *outside* the nearest core
microcluster, so the per-point outlier score is the distance to the best
microcluster center divided by that cluster's radius.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .enhancement import AntiHabituation
from .microcluster import ClusterSpace, MicroCluster
from .projection import FlyProjection, winner_take_all

__all__ = ["AHSC"]

_EPS = 1e-8


def _kmeans_pp_centers(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding followed by a few Lloyd iterations (Algorithm 2)."""
    n = len(X)
    k = min(k, n)
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            [np.sum((X - c) ** 2, axis=1) for c in centers], axis=0
        )
        total = d2.sum()
        probs = d2 / total if total > 0 else np.ones(n) / n
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)
    for _ in range(5):
        assign = np.argmin(
            np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2), axis=1
        )
        for c in range(k):
            members = X[assign == c]
            if len(members):
                C[c] = members.mean(axis=0)
    return C


class AHSC:
    """Anti-Drosophila Habituation Stream Clustering for anomaly detection."""

    def __init__(
        self,
        expansion: int = 40,
        density: float = 0.1,
        wta_frac: float = 0.05,
        alpha: float = 0.1,
        beta: float = 0.05,
        n_init: int = 8,
        theta: float = 100.0,
        delta: int = 2,
        core_fraction: float = 0.5,
        rebuild_every: int = 50,
        seed: int = 0,
    ) -> None:
        self.expansion = int(expansion)
        self.density = float(density)
        self.wta_frac = float(wta_frac)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.n_init = int(n_init)
        self.theta = float(theta)
        self.delta = int(delta)
        self.core_fraction = float(core_fraction)
        self.rebuild_every = int(rebuild_every)
        self.seed = int(seed)

        self._proj: FlyProjection | None = None
        self._enh_weights: np.ndarray | None = None  # frozen after fit
        self._init_radius = 1.0
        self.space: ClusterSpace | None = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # preprocessing
    # ------------------------------------------------------------------ #
    def _sparse_codes(self, X: np.ndarray) -> np.ndarray:
        """Project + winner-take-all (no habituation state)."""
        assert self._proj is not None
        Y = self._proj.transform(X)
        return winner_take_all(Y, self.wta_frac)

    def _encode_fit(self, X: np.ndarray) -> np.ndarray:
        """Streaming encode that learns the anti-habituation background."""
        S = self._sparse_codes(X)
        enh = AntiHabituation(dim=S.shape[1], alpha=self.alpha, beta=self.beta)
        E = enh.transform(S)
        self._enh_weights = enh.weights.copy()  # freeze for scoring
        return E

    def _encode_frozen(self, X: np.ndarray) -> np.ndarray:
        """Stateless encode using the frozen background (Eq. 8 only)."""
        assert self._enh_weights is not None
        S = self._sparse_codes(X)
        return np.maximum(S - self._enh_weights, 0.0)

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "AHSC":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        rng = np.random.default_rng(self.seed)
        self._proj = FlyProjection(
            input_dim=X.shape[1],
            expansion=self.expansion,
            density=self.density,
            seed=self.seed,
        )
        E = self._encode_fit(X)

        # A length scale for radii: typical spread of the encoded points.
        self._init_radius = float(np.median(np.linalg.norm(E - E.mean(0), axis=1))) + _EPS

        self.space = ClusterSpace(theta=self.theta, r_max=self._init_radius * 10.0, delta=self.delta)

        # Step 2: initialise microclusters from the first block with k-means++.
        init_n = min(len(E), max(self.n_init * 5, 50))
        init_block = E[:init_n]
        centers = _kmeans_pp_centers(init_block, self.n_init, rng)
        assign = np.argmin(
            np.linalg.norm(init_block[:, None, :] - centers[None, :, :], axis=2), axis=1
        )
        for c in range(len(centers)):
            count = int((assign == c).sum())
            if count == 0:
                continue
            self.space.add(
                MicroCluster(
                    center=centers[c].copy(),
                    radius=self._init_radius,
                    kn=count,
                    weight=1.0,
                    status="core" if count >= self.delta else "potential",
                )
            )
        self.space.build_macroclusters()

        # Steps 3-6: stream the rest of the points through the model.
        for i in range(init_n, len(E)):
            self._consume(E[i])
            if (i - init_n + 1) % self.rebuild_every == 0:
                self._evict()
                self.space.build_macroclusters()

        self._evict()
        self.space.build_macroclusters()
        self._fitted = True
        return self

    def _consume(self, y: np.ndarray) -> None:
        """Map one point into the model (steps 3-4)."""
        assert self.space is not None
        best, dist = self.space.find_best(y)
        if best is not None and dist <= best.radius:
            if dist <= self.core_fraction * best.radius:
                best.kn += 1  # core region (Eq. 13)
            else:
                best.update_center_shell(y)  # shell region (Eq. 4 / Eq. 14)
                best.update_radius(dist, self.theta, self.space.r_max)  # Eq. 1
            best.reinforce(dist, self.theta)  # Eq. 3 (energy)
            if best.is_core(self.delta):
                best.status = "core"
            self.space.touch()
        else:
            # No owning microcluster: spawn a new potential one.
            self.space.add(
                MicroCluster(
                    center=np.asarray(y, dtype=float).copy(),
                    radius=self._init_radius,
                    kn=1,
                    weight=1.0,
                    status="potential",
                )
            )

    def _evict(self) -> None:
        """Fade idle microclusters and remove the abnormal ones (step 5)."""
        assert self.space is not None
        survivors: List[MicroCluster] = []
        for mc in self.space.microclusters:
            mc.fade(self.theta)
            if mc.weight > 0.0:
                survivors.append(mc)
            elif mc.status == "core":
                mc.weight = 0.5  # demote to offline buffer, do not delete
                mc.status = "buffer"
                survivors.append(mc)
            # potential/buffer with weight <= 0 are dropped (abnormal).
        if len(survivors) != len(self.space.microclusters):
            self.space.microclusters = survivors
            self.space.touch()

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def outlier_scores(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("AHSC is not fitted; call fit(X) first")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        E = self._encode_frozen(X)
        scores = np.empty(len(E))
        for i, e in enumerate(E):
            best, dist = self.space.find_best(e)
            if best is None:
                scores[i] = dist
            else:
                scores[i] = dist / (best.radius + _EPS)
        return scores
