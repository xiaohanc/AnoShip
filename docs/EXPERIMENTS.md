# Experiments and reproduction

This repository accompanies our publications on time-series anomaly detection and
deployment safety. This guide explains how to **fully reproduce** the methods —
both the lightweight reference detectors built into `anoship` and the **exact
published model** (MSTDF-AD), which is vendored here verbatim.

---

## 1. Two levels of reproduction

| Goal | Use | Fidelity |
|------|-----|----------|
| Understand / run the published methods quickly, no datasets, no GPU | built-in detectors (`habituation`, `causal`, `diffusion`, `spatiotemporal`, `ewma`) | NumPy reference implementations (`habituation` is the full AHSC algorithm; the others capture the core idea) |
| Reproduce the **exact published model + numbers** | the vendored `mstdf_ad` package (`anoship-mstdf`) | byte-for-byte upstream model/training/eval |

The built-in detectors require only NumPy and run in seconds. The exact model
requires PyTorch and the benchmark datasets.

---

## 2. Quick reproduction — built-in detectors (no datasets)

```bash
bash scripts/dev_install.sh           # installs the 5 core packages (NumPy-only)
python examples/end_to_end.py         # detector x scenario outcome matrix
pytest -q                             # 55 tests
```

This demonstrates each method's behavior on reproducible synthetic scenarios
(healthy / regression / drift / spike / noisy) through the full deployment
pipeline (progressive rollout → health gate → automated rollback).

---

## 3. Exact reproduction — the real MSTDF-AD model

### 3.1 Install

```bash
pip install -e anoship-mstdf          # PyTorch, PyWavelets, etc.
```

### 3.2 Get the benchmark datasets

Download each dataset and place it under `anoship-mstdf/mstdf_ad/dataset/<NAME>/`:

| Dataset | Source |
|---------|--------|
| PSM     | https://github.com/eBay/RANSynCoders/tree/main/data |
| SMD     | https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset |
| SWaT    | request form: https://itrust.sutd.edu.sg/itrust-labs_datasets/ |
| MSL / SMAP | `https://s3-us-west-2.amazonaws.com/telemanom/data.zip` and `labeled_anomalies.csv` from https://github.com/khundman/telemanom |

Then build the pickled inputs (run inside each dataset folder):

```bash
cd anoship-mstdf/mstdf_ad/dataset/PSM && python make_pk.py
```

### 3.3 Train and evaluate

```bash
cd anoship-mstdf
python -m mstdf_ad.main --dataset PSM --epochs 64
```

This prints precision / recall / F1 for the dataset. Repeat with
`--dataset SMD|SWaT|MSL|SMAP`. This path is the **authoritative** reproduction —
the code under `mstdf_ad/` is the original implementation (only import paths were
re-rooted under the package). See [`../anoship-mstdf/NOTICE`](../anoship-mstdf/NOTICE)
for authorship and attribution.

---

## 4. Using the published model inside anoship

The same model is exposed as a pluggable detector so it can be selected in a
deployment pipeline:

```python
import anoship.app as ans
import anoship.contrib.mstdf       # registers the "mstdf" detector (requires torch)

scn = ans.build_scenario("regression")
pipe = ans.DeploymentPipeline(
    detector=ans.DETECTORS.create("mstdf", epochs=10),
    rollout=ans.ROLLOUTS.create("canary"),
    policy=ans.POLICIES.create("risk_aware"),
).fit(scn.baseline)
report = pipe.run(scn.source)
```

> The `mstdf` adapter targets live-stream deployment use; for exact benchmark
> numbers use the `mstdf_ad` pipeline in §3.

---

## 5. Citation

If you reproduce or build on these methods, please cite the corresponding
papers (see the publication list / paper-to-module table in the
[README](../README.md)).
