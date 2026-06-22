# Databricks vLLM Runtime Preflight Gate Evidence

This evidence covers PR #315, which wires Cachet's strict vLLM runtime
preflight into Databricks engine-probe job generation and execution for
release-safe provider-backed vLLM probes.

## Verification

- `python -m pytest -q tests/test_databricks_engine_probe_job.py tests/test_benchmark_plan.py`
- `poetry check`
- `python -m pytest -q`
- `git diff --check`
- changed-file credential-pattern scan
