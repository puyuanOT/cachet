# Engine Probe Runtime Packages PR Evidence

PR: https://github.com/puyuanOT/document-kv-cache/pull/299

## What Changed

- Added Databricks engine-probe runner support for `--pip-package` runtime package specs.
- Added single-target and matrix job config fields for extra PyPI package specs.
- Added per-backend `pip_packages` in engine-probe target JSON so vLLM and SGLang runtimes can install separately.
- Kept runtime package installation before local Cachet and adapter wheels.
- Mirrored the new validation in the legacy `restaurant_kv_serving` compatibility wrapper.

## Review

GPT-5.5 high-reasoning review found one P2 compatibility gap: the legacy wrapper bypassed the new pip-package validation. The PR was patched so legacy target, matrix, and single-job configs validate and tuple-coerce the new fields through the isolated document namespace. Delta review approved with no findings.

## Verification

- `PYTHONPATH=src pytest tests/test_databricks_engine_probe_job.py -q`
- `PYTHONPATH=src pytest tests/test_databricks_engine_probe_job.py tests/test_benchmark_plan.py tests/test_benchmark_plan_executor.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m compileall -q src tests`
- `git diff --check`
- `PYTHONPATH=src pytest -q`
