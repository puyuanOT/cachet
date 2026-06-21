# Databricks Probe Fixture Targets

This directory contains PR evidence for making Databricks engine-probe target
jobs self-sufficient when deterministic Qwen3 probe fixtures are generated as
part of the job.

Verification recorded here:

- `pytest -q tests/test_databricks_engine_probe_job.py tests/test_benchmark_plan.py tests/test_public_package.py`
- `pytest -q`
- `python -m build --wheel`

GPT-5.5 review found relative-path aliasing and unsupported URI-scheme risks.
The PR resolved them with exact derived fixture URI validation, supported-scheme
checks, and focused regression tests.
