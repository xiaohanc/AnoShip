# anoship-mstdf

The **real MSTDF-AD** spatiotemporal-fusion anomaly detector (PyTorch), packaged
two ways:

1. **`mstdf_ad/`** — the verbatim upstream implementation (model + training +
   evaluation + data pipeline), so the published method can be **reproduced in
   full**.
2. **`anoship.contrib.mstdf.MSTDFDetector`** — an adapter that exposes the model
   as a pluggable `anoship` detector (registered under the name `mstdf`), so it
   can be selected in a deployment pipeline like any other detector.

See [`NOTICE`](NOTICE) for authorship and attribution.

## Install

```bash
pip install -e anoship-mstdf   # pulls torch, PyWavelets, etc.
```

## Reproduce the published results

Use the vendored pipeline exactly as upstream (datasets prepared under
`mstdf_ad/dataset/<NAME>/`):

```bash
python -m mstdf_ad.main --dataset PSM --epochs 64
```

This path is byte-for-byte the original implementation (only import paths were
re-rooted under the `mstdf_ad` package); it is the authoritative reproduction.

## Use it as a pluggable anoship detector

```python
import anoship.app as ans
import anoship.contrib.mstdf  # registers the "mstdf" detector (requires torch)

scn = ans.build_scenario("regression")
pipe = ans.DeploymentPipeline(
    detector=ans.DETECTORS.create("mstdf", epochs=10),
    rollout=ans.ROLLOUTS.create("canary"),
    policy=ans.POLICIES.create("risk_aware"),
).fit(scn.baseline)
report = pipe.run(scn.source)
```

> **Note.** The `MSTDFDetector` adapter wires the real model into anoship's
> `(T, C)`-stream interface (it standardizes inputs, builds the calendar/`time`
> features and wavelet `S`/`T` components the model expects, trains with the
> upstream KL + reconstruction objective, and scores with the upstream
> reconstruction-error × KL-contribution formula). For **exact** paper numbers,
> use the `mstdf_ad` reproduction path above — the adapter targets live-stream
> deployment use, not benchmark reproduction.
