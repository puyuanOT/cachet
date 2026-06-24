# Centralize Hardware Targets PR Evidence

PR: https://github.com/puyuanOT/document-kv-cache/pull/298

## What Changed

- Added a private `_hardware_targets.py` profile for the V1 hardware target.
- Routed benchmark defaults, Databricks job defaults, and Databricks run-status validation through the shared profile.
- Kept legacy public validator aliases and module identities intact.
- Added focused tests for the shared Databricks prefix map and public wrapper compatibility.

## Review

GPT-5.5 high-reasoning review found no blocking issues and called out two residual risks: `databricks_runs.py` still had a hardcoded g6 prefix map, and direct validator imports could alter public `__module__` identity. Both were patched in this PR. Delta review approved the branch with no findings.

## Verification

- `PYTHONPATH=src pytest tests/test_benchmarks.py::test_benchmark_suite_defaults_to_v1_contract tests/test_databricks_job.py::test_generic_single_node_gpu_aliases_preserve_g5_compatibility_names tests/test_databricks_runs.py::test_databricks_run_status_uses_shared_hardware_target_prefixes -q`
- `PYTHONPATH=src pytest tests/test_benchmarks.py tests/test_databricks_job.py tests/test_databricks_runs.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m compileall -q src tests`
- `git diff --check`
- `PYTHONPATH=src pytest -q`
