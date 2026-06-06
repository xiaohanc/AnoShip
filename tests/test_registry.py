import numpy as np
import pytest
from anoship.core.errors import RegistryError
from anoship.core.registry import Registry


def test_register_and_create():
    reg = Registry("widget")

    @reg.register("foo")
    class Foo:
        def __init__(self, x=1):
            self.x = x

    assert "foo" in reg
    assert reg.names() == ["foo"]
    obj = reg.create("foo", x=5)
    assert obj.x == 5


def test_duplicate_registration_raises():
    reg = Registry("widget")
    reg.register("foo", type("A", (), {}))
    with pytest.raises(RegistryError):
        reg.register("foo", type("B", (), {}))


def test_unknown_name_raises():
    reg = Registry("widget")
    with pytest.raises(RegistryError):
        reg.get("missing")


def test_builtin_registries_populated():
    import anoship.app  # noqa: F401  triggers registration
    from anoship.core.registry import DETECTORS, POLICIES, ROLLOUTS, SCENARIOS

    for name in ["causal", "diffusion", "habituation", "spatiotemporal", "ewma"]:
        assert name in DETECTORS
    assert "canary" in ROLLOUTS
    assert "risk_aware" in POLICIES
    assert "regression" in SCENARIOS
