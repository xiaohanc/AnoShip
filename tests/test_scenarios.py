import anoship.app  # noqa: F401
import numpy as np
import pytest
from anoship.core.registry import SCENARIOS
from anoship.scenarios import build_scenario

SCENARIO_NAMES = ["healthy", "regression", "drift", "spike", "noisy"]


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_build_scenario_shapes(name):
    scn = build_scenario(name, n_channels=4, n_stages=5)
    assert scn.baseline.shape[1] == 4
    assert scn.name == name
    # Source must yield a 2-D window for each stage.
    for i in range(5):
        from anoship.core.types import RolloutStage

        w = scn.source.observe(RolloutStage(name=f"s{i}", exposure=0.5, index=i))
        assert w.ndim == 2 and w.shape[1] == 4


def test_all_scenarios_registered():
    for name in SCENARIO_NAMES:
        assert name in SCENARIOS


def test_scenarios_are_reproducible():
    a = build_scenario("regression", seed=3)
    b = build_scenario("regression", seed=3)
    assert np.allclose(a.baseline, b.baseline)
