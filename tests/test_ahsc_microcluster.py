import numpy as np
from anoship.detectors._ahsc.microcluster import ClusterSpace, MicroCluster


def test_density_is_core_plus_shell_counts():
    mc = MicroCluster(center=np.zeros(2), radius=1.0, kn=3, sn=2)
    assert mc.density == 5  # N = KN + SN  (Eq. 2)


def test_radius_update_follows_eq1():
    mc = MicroCluster(center=np.zeros(2), radius=1.0)
    mc.update_radius(dist=1.0, theta=2.0, r_max=10.0)
    # R + ((2d/R) - 1) * (1/theta) = 1 + (2-1)*0.5 = 1.5
    assert np.isclose(mc.radius, 1.5)


def test_radius_update_capped_at_rmax():
    mc = MicroCluster(center=np.zeros(2), radius=1.0)
    mc.update_radius(dist=100.0, theta=1.0, r_max=3.0)
    assert mc.radius == 3.0


def test_shell_center_update_is_running_mean_eq4():
    mc = MicroCluster(center=np.zeros(2), radius=1.0)
    mc.update_center_shell(np.array([2.0, 2.0]))
    assert np.allclose(mc.center, [2.0, 2.0]) and mc.sn == 1
    mc.update_center_shell(np.array([4.0, 4.0]))
    assert np.allclose(mc.center, [3.0, 3.0]) and mc.sn == 2


def test_reinforce_rewards_closer_points_more():
    near = MicroCluster(center=np.zeros(2), radius=2.0, weight=1.0)
    far = MicroCluster(center=np.zeros(2), radius=2.0, weight=1.0)
    near.reinforce(dist=0.2, theta=1.0)
    far.reinforce(dist=1.8, theta=1.0)
    assert near.weight > far.weight > 1.0


def test_fade_decreases_weight():
    mc = MicroCluster(center=np.zeros(2), radius=1.0, weight=1.0)
    mc.fade(theta=2.0)
    assert np.isclose(mc.weight, 0.5)


def test_is_core_requires_density_and_positive_weight():
    mc = MicroCluster(center=np.zeros(2), radius=1.0, kn=4, sn=1, weight=0.5)
    assert mc.is_core(delta=3)
    assert not mc.is_core(delta=10)  # too few points
    mc.weight = -0.1
    assert not mc.is_core(delta=3)  # decayed below zero


def test_build_macroclusters_groups_overlapping_microclusters():
    a = MicroCluster(center=np.array([0.0, 0.0]), radius=1.0)
    b = MicroCluster(center=np.array([1.5, 0.0]), radius=1.0)  # overlaps a
    c = MicroCluster(center=np.array([20.0, 0.0]), radius=1.0)  # far away
    space = ClusterSpace(theta=1.0, r_max=5.0, delta=1)
    space.microclusters = [a, b, c]
    space.build_macroclusters()
    assert a.macro_id == b.macro_id
    assert a.macro_id != c.macro_id
    assert len(space.macro_centers()) == 2


def test_macro_center_is_mean_of_members():
    a = MicroCluster(center=np.array([0.0, 0.0]), radius=1.0)
    b = MicroCluster(center=np.array([2.0, 0.0]), radius=2.0)
    space = ClusterSpace(theta=1.0, r_max=5.0, delta=1)
    space.microclusters = [a, b]
    space.build_macroclusters()
    centers = space.macro_centers()
    assert np.allclose(centers[a.macro_id], [1.0, 0.0])


def test_macro_first_search_matches_brute_force_when_separated():
    rng = np.random.default_rng(0)
    space = ClusterSpace(theta=1.0, r_max=5.0, delta=1)
    # three well-separated tight groups
    centers = [np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, 10.0])]
    space.microclusters = [MicroCluster(center=c.copy(), radius=0.5) for c in centers]
    space.build_macroclusters()
    for _ in range(20):
        x = centers[rng.integers(3)] + rng.normal(scale=0.1, size=2)
        best, dist = space.find_best(x)
        brute = min(space.microclusters, key=lambda m: np.linalg.norm(x - m.center))
        assert best is brute
