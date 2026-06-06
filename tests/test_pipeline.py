import anoship.app as ans
import numpy as np
import pytest
from anoship.core.context import RunContext
from anoship.core.errors import NotFittedError, PipelineError, RollbackError
from anoship.core.types import DeploymentState
from anoship.pipeline.state_machine import DeploymentStateMachine
from anoship.snapshots.registry import SnapshotRegistry


def _pipeline(
    detector="diffusion", rollout="canary", policy="risk_aware", tier="standard"
):
    return ans.DeploymentPipeline(
        detector=ans.DETECTORS.create(detector),
        rollout=ans.ROLLOUTS.create(rollout),
        policy=ans.POLICIES.create(policy),
        risk_tier=tier,
    )


def test_run_requires_fit():
    scn = ans.build_scenario("healthy")
    with pytest.raises(NotFittedError):
        _pipeline().run(scn.source)


def test_healthy_completes():
    scn = ans.build_scenario("healthy")
    report = _pipeline().fit(scn.baseline).run(scn.source)
    assert report.success
    assert not report.rolled_back
    assert report.summary.stages_promoted == report.summary.stages_total


def test_regression_rolls_back_with_root_cause():
    scn = ans.build_scenario("regression")
    report = _pipeline(tier="high_impact").fit(scn.baseline).run(scn.source)
    assert report.rolled_back
    assert report.summary.detection_delay is not None
    assert report.root_cause is not None
    # Perturbed channels were 0 and 1.
    assert report.root_cause.top_channel in {"ch0", "ch1"}


def test_noisy_not_rolled_back_with_diffusion():
    scn = ans.build_scenario("noisy")
    report = _pipeline(detector="diffusion").fit(scn.baseline).run(scn.source)
    assert not report.rolled_back


def test_rollback_restores_safe_snapshot():
    scn = ans.build_scenario("spike")
    report = _pipeline().fit(scn.baseline).run(scn.source)
    assert report.rolled_back
    assert report.context.snapshots.safe_snapshot.snapshot_id == "baseline-safe"
    assert report.context.state == DeploymentState.ROLLED_BACK


def test_illegal_state_transition():
    ctx = RunContext()
    sm = DeploymentStateMachine(ctx)
    with pytest.raises(PipelineError):
        sm.transition(DeploymentState.COMPLETED)  # PENDING -> COMPLETED illegal


def test_snapshot_registry_rollback_target():
    reg = SnapshotRegistry()
    with pytest.raises(RollbackError):
        reg.rollback_target()
    reg.register("safe", safe=True)
    assert reg.rollback_target().snapshot_id == "safe"
