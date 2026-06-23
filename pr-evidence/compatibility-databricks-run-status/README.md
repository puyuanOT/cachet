# Compatibility Databricks Run Status

This folder contains PR evidence for adding compatibility Databricks run-status
evidence to strict release bundles.

The PR adds a `compatibility_databricks_run_status` artifact role for
non-default compatibility benchmark targets such as AWS g5/A10G. The role keeps
compatibility status evidence separate from the strict AWS g6/L4 release target,
rejects default-target statuses, requires the status target to match a bundled
compatibility benchmark, and requires compatibility statuses to be V1
benchmark/vLLM-smoke statuses rather than storage or engine-probe statuses.

Verification:

- `git diff --check`
- `python -m py_compile src/document_kv_cache/release_bundle.py src/document_kv_cache/benchmark_plan.py`
- `poetry check && poetry check --lock`
- `poetry run pytest tests/test_release_bundle.py tests/test_benchmark_plan.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- strict 23-artifact g5-enriched release bundle rebuild with the current g5
  compatibility run-status sidecar
- GPT-5.5 focused review with findings resolved and approved
