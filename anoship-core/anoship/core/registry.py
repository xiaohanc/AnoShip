"""Plugin registry: the mechanism that makes anoship components discoverable.

Components register themselves under a string name, and the CLI / config layer
instantiates them by name. This is how a YAML deployment spec like
``detector: causal`` is resolved into a concrete object, and how external
organizations can contribute new detectors without modifying the core.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar

from .errors import RegistryError

__all__ = [
    "Registry",
    "DETECTORS",
    "ROLLOUTS",
    "POLICIES",
    "THRESHOLDS",
    "EXPORTERS",
    "SCENARIOS",
    "register_detector",
    "register_rollout",
    "register_policy",
    "register_threshold",
    "register_exporter",
    "register_scenario",
]

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection of pluggable classes of a single kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, Type[T]] = {}

    def register(
        self, name: str, obj: Optional[Type[T]] = None
    ) -> Callable[[Type[T]], Type[T]] | Type[T]:
        """Register ``obj`` under ``name``.

        Usable both as a decorator (``@reg.register("foo")``) and as a direct
        call (``reg.register("foo", Foo)``).
        """

        def _do(target: Type[T]) -> Type[T]:
            key = name.lower()
            if key in self._items:
                raise RegistryError(f"{self.kind} {name!r} is already registered")
            self._items[key] = target
            return target

        if obj is not None:
            return _do(obj)
        return _do

    def get(self, name: str) -> Type[T]:
        key = name.lower()
        if key not in self._items:
            raise RegistryError(
                f"unknown {self.kind} {name!r}; available: {', '.join(self.names()) or '(none)'}"
            )
        return self._items[key]

    def create(self, name: str, *args, **kwargs) -> T:
        """Instantiate the registered class ``name`` with the given args."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)


# Global registries, one per extension point.
DETECTORS: Registry = Registry("detector")
ROLLOUTS: Registry = Registry("rollout strategy")
POLICIES: Registry = Registry("gate policy")
THRESHOLDS: Registry = Registry("threshold strategy")
EXPORTERS: Registry = Registry("exporter")
SCENARIOS: Registry = Registry("scenario")


def register_detector(name: str):
    return DETECTORS.register(name)


def register_rollout(name: str):
    return ROLLOUTS.register(name)


def register_policy(name: str):
    return POLICIES.register(name)


def register_threshold(name: str):
    return THRESHOLDS.register(name)


def register_exporter(name: str):
    return EXPORTERS.register(name)


def register_scenario(name: str):
    return SCENARIOS.register(name)
