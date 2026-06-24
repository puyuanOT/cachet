# Normalize Baseline Prefill Labels

## What Changed

- Updated the V1 requirements matrix to name the canonical `baseline_prefill`
  no-cache arm.
- Updated the strict release-bundle requirements-matrix snippet guard and the
  governance test to expect the same arm label.

## Why

The benchmark runner, release-evidence validator, and Databricks vLLM smoke
artifact use `baseline_prefill`. Removing the stale `full_no_cache` wording keeps
release-gate documentation aligned with the machine-checked benchmark schema.

## Scope

- `docs/v1-requirements-matrix.md`
- `src/document_kv_cache/release_bundle.py`
- `tests/test_project_governance.py`

## Verification

- `pytest -q tests/test_project_governance.py tests/test_release_bundle.py` -> 99 passed
- `pytest -q` -> 1265 passed
- `python -m build --wheel` -> built `document_kv_cache-0.2.0-py3-none-any.whl`
- GPT-5.5 review -> approved with no findings
