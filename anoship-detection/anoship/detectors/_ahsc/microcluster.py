"""Evolving micro/macro-cluster structure (AHSC, Sec. II-III).

AHSC summarises the stream as two nested structures:

* **microcluster** -- a synopsis ``(KN, SN, C, R, W)``: counts of core/shell
  points, a center, a dynamically updated radius, and a fading weight. A
  microcluster never stores raw points.
* **macrocluster** -- a connected group of cross-linked microclusters (centers
  within overlapping radii). Macroclusters give the *macrocluster-first* lookup
  that cuts the cost of locating a new point's best microcluster from O(n) to
  roughly O(log n).

Microclusters cycle through three states by their weight ``W`` and density
``N = KN + SN``: *core* (``N >= delta`` and ``W > 0``), *potential* (``W > 0`` but
too sparse to be core yet), and *offline-buffer* (a former core whose weight has
decayed; reactivated if it is hit again).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["MicroCluster", "ClusterSpace"]


@dataclass
class MicroCluster:
    """A single microcluster synopsis."""

    center: np.ndarray
    radius: float
    kn: int = 0  # core-region point count
    sn: int = 0  # shell-region point count
    weight: float = 1.0
    macro_id: Optional[int] = None
    status: str = "potential"  # one of: core | potential | buffer

    @property
    def density(self) -> int:
        """Local density ``N = KN + SN`` (Eq. 2)."""
        return int(self.kn + self.sn)

    def update_radius(self, dist: float, theta: float, r_max: float) -> None:
        """Dynamic radius update (Eq. 1)."""
        r = self.radius
        r_next = r + ((2.0 * dist / r) - 1.0) * (1.0 / theta)
        self.radius = float(min(r_next, r_max))

    def update_center_shell(self, x: np.ndarray) -> None:
        """Add a shell point and update the center as a running mean (Eq. 4)."""
        self.sn += 1
        self.center = self.center + (np.asarray(x, dtype=float) - self.center) / self.sn

    def reinforce(self, dist: float, theta: float) -> None:
        """Reward a hit; closer points add more energy (Eq. 3, energy term)."""
        self.weight += max(self.radius - dist, 0.0) / self.radius / theta

    def fade(self, theta: float) -> None:
        """Decay the weight of a microcluster that received no update (Eq. 3)."""
        self.weight -= 1.0 / theta

    def is_core(self, delta: int) -> bool:
        """A core microcluster has enough points and a positive weight (Def. 3)."""
        return self.density >= delta and self.weight > 0.0


class ClusterSpace:
    """Container of microclusters plus the macrocluster graph over them."""

    def __init__(self, theta: float, r_max: float, delta: int) -> None:
        self.theta = float(theta)
        self.r_max = float(r_max)
        self.delta = int(delta)
        self.microclusters: List[MicroCluster] = []
        self._macro_centers: List[np.ndarray] = []
        self._dirty = True  # macrocluster graph needs (re)building

    # ------------------------------------------------------------------ #
    # membership (online loop support)
    # ------------------------------------------------------------------ #
    def add(self, mc: MicroCluster) -> None:
        self.microclusters.append(mc)
        self._dirty = True

    def remove(self, mc: MicroCluster) -> None:
        self.microclusters.remove(mc)
        self._dirty = True

    def touch(self) -> None:
        """Flag the macrocluster graph as stale after in-place edits."""
        self._dirty = True

    # ------------------------------------------------------------------ #
    # macrocluster graph
    # ------------------------------------------------------------------ #
    def build_macroclusters(self) -> None:
        """Group cross-connected microclusters into macroclusters.

        Two microclusters are cross-connected when their centers lie within the
        sum of their radii (overlapping spheres). Connected components become
        macroclusters; each macrocluster center ``GC`` is the mean of its member
        centers.
        """
        mcs = self.microclusters
        n = len(mcs)
        for mc in mcs:
            mc.macro_id = None
        next_id = 0
        for i in range(n):
            if mcs[i].macro_id is not None:
                continue
            # BFS over the overlap graph starting from i.
            mcs[i].macro_id = next_id
            queue = [i]
            while queue:
                u = queue.pop()
                for v in range(n):
                    if mcs[v].macro_id is not None:
                        continue
                    gap = float(np.linalg.norm(mcs[u].center - mcs[v].center))
                    if gap <= mcs[u].radius + mcs[v].radius:
                        mcs[v].macro_id = next_id
                        queue.append(v)
            next_id += 1

        self._macro_centers = []
        for g in range(next_id):
            members = [m.center for m in mcs if m.macro_id == g]
            self._macro_centers.append(np.mean(members, axis=0))
        self._dirty = False

    def macro_centers(self) -> List[np.ndarray]:
        return self._macro_centers

    # ------------------------------------------------------------------ #
    # macrocluster-first search (Eq. 10-12)
    # ------------------------------------------------------------------ #
    def find_best(self, x: np.ndarray) -> Tuple[Optional[MicroCluster], float]:
        """Locate the best microcluster for ``x`` via macrocluster-first search.

        Picks the nearest macrocluster center ``GC`` (Eq. 10-11), then the nearest
        microcluster within that macrocluster. Returns ``(microcluster, distance)``
        or ``(None, inf)`` when the space is empty.
        """
        if not self.microclusters:
            return None, float("inf")
        x = np.asarray(x, dtype=float)
        if self._dirty or not self._macro_centers:
            self.build_macroclusters()

        # Eq. 10-11: nearest macrocluster.
        gdists = [float(np.linalg.norm(x - gc)) for gc in self._macro_centers]
        best_macro = int(np.argmin(gdists))

        # Nearest microcluster inside the chosen macrocluster.
        best: Optional[MicroCluster] = None
        best_dist = float("inf")
        for mc in self.microclusters:
            if mc.macro_id != best_macro:
                continue
            d = float(np.linalg.norm(x - mc.center))
            if d < best_dist:
                best_dist = d
                best = mc
        return best, best_dist
