# Architecture

`anoship` is organized as a set of layers with a single, stable vocabulary of
core types flowing between them. Each layer depends only on the layers below it,
and every extension point is a registered plugin.

```
┌─────────────────────────────────────────────────────────────┐
│  CLI / Config / Reporting        (entry points, YAML, reports)│
├─────────────────────────────────────────────────────────────┤
│  Orchestration Layer        (pipeline)                        │
│   pipeline · rollout strategies · health gate · rollback      │
│   promotion · deployment state machine                        │
├─────────────────────────────────────────────────────────────┤
│  Policy & Governance Layer  (policy)                          │
│   risk tiers · gate policies · decision engine                │
├─────────────────────────────────────────────────────────────┤
│  Detection Layer            (detectors · scoring · attribution)│
│   detectors (one per paper) · ensemble · thresholds           │
│   score fusion · calibration · root-cause attribution         │
├─────────────────────────────────────────────────────────────┤
│  Signal Layer               (signals)                         │
│   sliding windows · typed channels · normalization · synthetic │
├─────────────────────────────────────────────────────────────┤
│  Observability Layer        (observability)                   │
│   event bus · metrics · exporters (console/json/null)         │
├─────────────────────────────────────────────────────────────┤
│  Core / Plugin Framework    (core)                            │
│   interfaces · registry · config schema · types · context     │
├─────────────────────────────────────────────────────────────┤
│  Adapters & Integrations    (adapters)                        │
│   PyTorch (MSTDF-AD) · scikit-learn                            │
└─────────────────────────────────────────────────────────────┘
```

## Extension points (interfaces)

All defined in `anoship/core/interfaces.py`; all resolved by name through
`anoship/core/registry.py`.

| Interface          | Responsibility                               | Built-ins |
|--------------------|----------------------------------------------|-----------|
| `Detector`         | `fit` / `score` / `is_anomaly` over windows  | `ewma`, `habituation`, `causal`, `diffusion`, `spatiotemporal`, `ensemble` |
| `ThresholdStrategy`| score → decision threshold                   | `static`, `sigma`, `quantile`, `adaptive` |
| `RolloutStrategy`  | how a snapshot is progressively exposed      | `canary`, `percentage`, `regional`, `blue_green` |
| `GatePolicy`       | anomaly result → promote/hold/rollback       | `threshold`, `risk_aware` |
| `Exporter`         | receive observability events                 | `console`, `json`, `null` |

## Core types

`anoship/core/types.py` defines the shared value objects exchanged between
layers: `AnomalyResult`, `RolloutStage`, `GateDecision`, `GateAction`,
`DeploymentState`, `Snapshot`, `Event`. Keeping these dependency-free (NumPy
only) is what lets the layers stay decoupled.

## Deployment data flow

1. **Fit.** `DeploymentPipeline.fit(baseline)` calibrates the detector (and its
   threshold) on anomaly-free data.
2. **Run.** For each `RolloutStage` from the `RolloutStrategy`:
   - The `SignalSource` yields the observed window for that stage.
   - `HealthGate` runs `detector.is_anomaly(window)` → `AnomalyResult`.
   - `GatePolicy` (risk-tier aware) maps the result to a `GateAction`:
     - `PROMOTE` → `PromotionController` advances to the next stage.
     - `HOLD` → promotion pauses (elevated but unconfirmed signal).
     - `ROLLBACK` → `RollbackController` restores the safe snapshot from the
       `SnapshotRegistry` and isolates the failure to the current stage/region.
   - Every step publishes a structured `Event` on the `EventBus`.
3. **Report.** A `RunReport` aggregates the final state, per-stage decisions, the
   full event timeline, metrics, and root-cause attribution.

## Decision logic & noise robustness

`anoship/policy/decision.py` is the heart of gating. Beyond a raw threshold it
uses **persistence**: it rolls back only on a *sustained run* of anomalous points
combined with either a rate breach or a severe excursion. Intermittent spikes
(characteristic of noisy telemetry) therefore do not trigger false rollbacks.
Risk tiers (`anoship/policy/risk_tier.py`) tighten these thresholds for
higher-impact deployments.

## Adapters

The detector interface is deliberately small so external models slot in:

- `adapters/torch_mstdf.py` wraps any reconstruction-based `torch.nn.Module`
  (e.g. the companion **MSTDF-AD** PyTorch model) as a `Detector`.
- `adapters/sklearn.py` wraps scikit-learn outlier/novelty estimators.

Both import their heavy dependencies lazily, so the core stays NumPy-only.
