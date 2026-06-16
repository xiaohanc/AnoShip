# anoship-scid

A **from-paper** PyTorch implementation of **SCID** (Spatiotemporal Causal
Inference Detector), packaged two ways:

1. **`scid_ad/`** — the model implemented *from the paper* (Knowledge-Based
   Systems 2025): the Predictive Reconstruction Process (PRP) dual-mask decoder,
   the Dynamic Causal Representation Encoder (DCRE) with counterfactual
   reasoning, and the dual-objective loss (MMD counterfactual differentiation +
   soft-DTW granularity adjustment) trained in two phases.
2. **`anoship.contrib.scid.SCIDDetector`** — an adapter that exposes the model
   as a pluggable `anoship` detector (registered under the name `scid`), so it
   can be selected in a deployment pipeline like any other detector.

See [`NOTICE`](NOTICE) for authorship and attribution.

> **No reproduction claim.** There is no public reference implementation of
> SCID, and the paper's benchmark datasets are not bundled here. This code is an
> independent interpretation of the paper and **does not claim to reproduce the
> paper's reported F1 numbers**. It is validated behaviorally on synthetic
> anomaly injection only.

## Install

```bash
pip install -e anoship-scid   # pulls torch, scipy, scikit-learn, pandas
```

## Use it as a pluggable anoship detector

```python
import anoship.app as ans
import anoship.contrib.scid  # registers the "scid" detector (requires torch)

scn = ans.build_scenario("regression")
pipe = ans.DeploymentPipeline(
    detector=ans.DETECTORS.create("scid"),   # quick preset by default
    rollout=ans.ROLLOUTS.create("canary"),
    policy=ans.POLICIES.create("risk_aware"),
).fit(scn.baseline)
report = pipe.run(scn.source)
```

The default configuration is a **quick preset** (small dims, few epochs) so the
detector fits in seconds for tests and interactive use. A documented
paper-scale configuration is available via `scid_ad.config.SCIDConfig`.

### Optional EVT thresholding

The detector defaults to anoship's framework-native `SigmaThreshold`. A
paper-faithful Peaks-Over-Threshold (POT / EVT) strategy is available:

```python
from scid_ad.pot import POTThreshold
det = ans.DETECTORS.create("scid", threshold=POTThreshold())
```
