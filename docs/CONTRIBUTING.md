# Contributing to anoship

Contributions are welcome — especially new anomaly-detection methods and rollout
strategies. The framework is built around a plugin registry so most extensions
require no changes to the core.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Adding a detector

1. Create `anoship/detectors/your_method.py`.
2. Subclass `BaseDetector` and implement two hooks:

   ```python
   import numpy as np
   from ..core.registry import register_detector
   from .base import BaseDetector

   @register_detector("your_method")
   class YourDetector(BaseDetector):
       name = "your_method"

       def _fit(self, X: np.ndarray) -> None:
           ...  # learn baseline state from normalized X (T, C)

       def _score(self, X: np.ndarray) -> np.ndarray:
           ...  # return per-row anomaly scores, shape (T,)
   ```

   `BaseDetector` handles normalization, thresholding, and root-cause
   attribution for you. Override `_attribution` if your method produces a
   stronger causal signal.

3. Export it in `anoship/detectors/__init__.py`.
4. Add a test in `tests/test_detectors.py` (it is parametrized over detector
   names — add yours to the list).

If the detector implements a published method, cite the paper in the module
docstring and add a row to the README's paper → module table.

## Adding a rollout strategy / gate policy / exporter

Subclass the corresponding interface in `anoship/core/interfaces.py` and register
it with `register_rollout` / `register_policy` / `register_exporter`. It will be
discoverable from the CLI and YAML configs immediately.

## Style & tests

- Keep the **core** dependency-free (NumPy only); put heavy deps behind adapters.
- Every new component needs a unit test; pipeline-affecting changes need an
  integration test in `tests/test_pipeline.py`.
- Run `pytest -q` before opening a PR.

## Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/).
Each commit message starts with a type and an optional scope:

```
<type>(<scope>): <short imperative summary>
```

Allowed **types**:

| Type       | Use for |
|------------|---------|
| `feat`     | a new feature or capability |
| `fix`      | a bug fix |
| `refactor` | a code change that neither fixes a bug nor adds a feature |
| `perf`     | a performance improvement |
| `test`     | adding or updating tests |
| `docs`     | documentation only |
| `build`    | packaging / build system (pyproject, dependencies) |
| `ci`       | CI configuration |
| `chore`    | tooling, ignores, housekeeping |
| `style`    | formatting only (no logic change) |

Common **scopes**: `core`, `signals`, `detection`, `pipeline`, `app`.

Examples:

```
feat(detection): add diffusion-denoising detector
refactor(pipeline): extract gate decision logic into policy.decision
build(core): scaffold anoship-core package
docs: add architecture overview
```
