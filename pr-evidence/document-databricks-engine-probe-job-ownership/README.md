# Document Databricks Engine Probe Job Ownership

This PR-evidence sidecar covers the refactor slice that moves the Databricks
`runs/submit` payload helper for native engine KV-connector probes into
`document_kv_cache.databricks_engine_probe_job`.

The slice makes the document package the implementation owner for the engine
probe job config, matrix target file parsing, single-node g5 cluster payload
generation, runner script emission, and CLI entry point. The legacy
`restaurant_kv_serving.databricks_engine_probe_job` module remains as a
compatibility wrapper.

## Review

GPT-5.5 reviewed the ownership inversion and found no code-level behavioral
regressions. It checked document and legacy monkeypatch isolation, direct config
construction, matrix target parsing, release-safe validation, legacy
`__all__`/star-import compatibility, pickle/slotted behavior, and recursion or
document-hook leak risks.

The only finding was that this sidecar itself was still marked pending, which
correctly failed the repository PR-evidence gate before review completion. The
sidecar now records the completed review outcome.

## Verification

- `poetry run pytest tests/test_databricks_engine_probe_job.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/databricks_engine_probe_job.py src/restaurant_kv_serving/databricks_engine_probe_job.py tests/test_databricks_engine_probe_job.py tests/test_public_package.py`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- `poetry check`
- `poetry build`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- GPT-5.5 review of the ownership migration with no code-level findings
