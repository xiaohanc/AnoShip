"""anoship.app -- high-level convenience API, builder, and CLI.

Importing this module wires up all plugin registries and exposes the framework's
entry points behind one import, so applications don't need to know which
distribution provides which component:

    import anoship.app as ans

    scn = ans.build_scenario("regression")
    pipe = ans.DeploymentPipeline(
        detector=ans.DETECTORS.create("diffusion"),
        rollout=ans.ROLLOUTS.create("canary"),
        policy=ans.POLICIES.create("risk_aware"),
        risk_tier="high_impact",
    ).fit(scn.baseline)
    report = pipe.run(scn.source)

In the monorepo, ``anoship`` is a PEP 420 namespace shared by several
distributions (anoship-core, anoship-signals, anoship-detection,
anoship-pipeline, anoship-app); this module lives in anoship-app and depends on
the rest.
"""

from __future__ import annotations

# Importing these subpackages registers their components in the global
# registries, so DETECTORS.create("causal") etc. work after `import anoship.app`.
from anoship import (  # noqa: F401  # noqa: F401
    detectors as _detectors,
    scenarios as _scenarios,
)
from anoship.core import (
    AnomalyResult,
    DeploymentConfig,
    DETECTORS,
    EXPORTERS,
    GateDecision,
    POLICIES,
    ROLLOUTS,
    RunContext,
    SCENARIOS,
    THRESHOLDS,
)
from anoship.observability import exporters as _exporters  # noqa: F401
from anoship.pipeline import (  # noqa: F401
    DeploymentPipeline,
    rollout as _rollout,
    RunReport,
    StaticSource,
)
from anoship.policy import gate_policy as _gate_policy  # noqa: F401
from anoship.scenarios import build_scenario
from anoship.scoring import thresholds as _thresholds  # noqa: F401

from .builder import build_detector, build_pipeline, build_threshold

__version__ = "0.1.0"

__all__ = [
    "DeploymentPipeline",
    "RunReport",
    "StaticSource",
    "build_scenario",
    "build_pipeline",
    "build_detector",
    "build_threshold",
    "DeploymentConfig",
    "RunContext",
    "AnomalyResult",
    "GateDecision",
    "DETECTORS",
    "ROLLOUTS",
    "POLICIES",
    "THRESHOLDS",
    "EXPORTERS",
    "SCENARIOS",
    "__version__",
]
