#!/usr/bin/env bash
# Install every anoship monorepo package in editable mode (dependency order).
# Cross-package deps (anoship-core etc.) aren't on PyPI, so we install with
# --no-deps and provide the third-party deps explicitly.
set -euo pipefail

python -m pip install -U pip
python -m pip install "numpy>=1.21" pyyaml pytest

for pkg in anoship-core anoship-signals anoship-detection anoship-pipeline anoship-app; do
    echo ">>> installing $pkg (editable)"
    python -m pip install -e "$pkg" --no-deps
done

echo "anoship monorepo installed in editable mode."
