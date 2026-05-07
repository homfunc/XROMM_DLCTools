# Contributing
This document covers local development, regression validation, and cross-repo integration checks for `XROMM_DLCTools`.

## Repository layout for integration work
The recommended local layout for cross-repo validation is:
- `XROMM_DLCTools/`
- `DeepLabCut/`
- `xmalab/`

The broader integration harness assumes this sibling layout when you use the default repo paths.

## Development environment
For lightweight development that does not need DeepLabCut imports:

```bash
uv lock
uv sync --no-group dlc
```

For DeepLabCut-dependent commands and full integration checks:

```bash
uv sync --group dlc
```

## Common commands
Inspect the packaged command surface:

```bash
uv run python -m xrommtools --help
```

Syntax and import sanity:

```bash
uv run python -m py_compile functions/xrommtools.py xrommtools/*.py
uv run python -c "from functions.xrommtools import xma_to_dlc, analyze_xromm_videos, dlc_to_xma, add_frames; print('import-ok')"
```

## Baseline harness
CI-safe baseline suite:

```bash
uv run python scripts/baseline_harness.py --scenario ci --output-dir baseline_artifacts/ci --fail-on-skip
```

Core regression suite:

```bash
uv run python scripts/baseline_harness.py --scenario core --output-dir baseline_artifacts/core
```

Full sibling-repo integration suite:

```bash
uv run python scripts/baseline_harness.py --scenario all --output-dir baseline_artifacts/integration_all --deeplabcut-repo ../DeepLabCut --xmalab-repo ../xmalab
```

## Golden manifest checks
Freeze the CI-safe baseline manifest:

```bash
uv run python scripts/baseline_manifest.py freeze --summary baseline_artifacts/ci/summary.json --out baseline_artifacts/golden/v1_ci_manifest.json --version v1
```

Verify the CI-safe manifest:

```bash
uv run python scripts/baseline_manifest.py verify --summary baseline_artifacts/ci/summary.json --manifest baseline_artifacts/golden/v1_ci_manifest.json
```
