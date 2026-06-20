# Release Evidence Action Sidecars

This PR makes engine connector action descriptors first-class release evidence.
Release evidence, preflight inspection, and release bundle assembly now require
one validated action sidecar per release backend alongside the native engine
probe records for vLLM and SGLang.

Benchmark plans can now emit `actions_output_json` for planned engine probes,
wire those sidecars into release evidence and release bundles, and carry the
same path through Databricks engine-probe target records. The legacy
`restaurant_kv_serving` CLIs pass the new arguments through to the
`document_kv_cache` implementations.

Verification:

- `poetry run pytest tests/test_benchmark_plan.py -q`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest tests/test_databricks_engine_probe_job.py tests/test_public_package.py -q`
- `python -m compileall -q src/document_kv_cache src/restaurant_kv_serving`
- `poetry run pytest -q`
- `git diff --check`

GPT-5.5 review approved the diff with no findings. The review did not rerun
live/native vLLM, SGLang, or Databricks jobs.
