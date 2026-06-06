import numpy as np
import pytest

from anoship.core.types import AnomalyResult, GateAction, RolloutStage
from anoship.policy.decision import decide, longest_run
from anoship.policy.risk_tier import TIERS, get_tier
from anoship.policy.gate_policy import RiskAwareGatePolicy, ThresholdGatePolicy
from anoship.core.errors import PolicyError


def _result(scores, threshold):
    scores = np.asarray(scores, dtype=float)
    mask = scores > threshold
    return AnomalyResult(
        scores=scores,
        score=float(scores.max()),
        threshold=threshold,
        is_anomaly=bool(mask.any()),
        anomaly_rate=float(mask.mean()),
    )


def test_longest_run():
    assert longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3
    assert longest_run([0, 0, 0]) == 0


def test_decide_promote_on_clean():
    res = _result(np.zeros(50), threshold=1.0)
    action, _ = decide(
        res, max_anomaly_rate=0.12, min_persistence=3, severity_ratio=2.5
    )
    assert action == GateAction.PROMOTE


def test_decide_rollback_on_sustained():
    scores = np.concatenate([np.zeros(30), np.full(20, 5.0)])
    res = _result(scores, threshold=1.0)
    action, reason = decide(
        res, max_anomaly_rate=0.12, min_persistence=3, severity_ratio=2.5
    )
    assert action == GateAction.ROLLBACK


def test_decide_no_rollback_on_spiky_noise():
    # Intermittent single-point exceedances: high-ish rate but no run.
    scores = np.zeros(50)
    scores[::4] = 5.0  # every 4th point, never consecutive
    res = _result(scores, threshold=1.0)
    action, _ = decide(
        res, max_anomaly_rate=0.12, min_persistence=3, severity_ratio=2.5
    )
    assert action != GateAction.ROLLBACK


def test_risk_tiers_monotonic():
    assert (
        TIERS["high_impact"].max_anomaly_rate
        < TIERS["standard"].max_anomaly_rate
        < TIERS["experimental"].max_anomaly_rate
    )


def test_get_tier_unknown():
    with pytest.raises(PolicyError):
        get_tier("nope")


def test_gate_policies_emit_decision():
    stage = RolloutStage(name="s", exposure=0.5, index=0)
    scores = np.concatenate([np.zeros(30), np.full(20, 5.0)])
    res = _result(scores, threshold=1.0)

    class Ctx:
        risk_tier = "high_impact"

    d1 = ThresholdGatePolicy().evaluate(res, stage, Ctx())
    d2 = RiskAwareGatePolicy().evaluate(res, stage, Ctx())
    assert d1.action == GateAction.ROLLBACK
    assert d2.action == GateAction.ROLLBACK
    assert "tier=high_impact" in d2.reason
