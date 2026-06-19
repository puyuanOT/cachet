# Document Databricks vLLM Smoke Job Ownership

This PR-evidence sidecar covers the refactor slice that moves the Databricks
`runs/submit` payload helper for the Qwen3/vLLM smoke benchmark into
`document_kv_cache.databricks_vllm_smoke_job`.

The slice makes the document package the implementation owner for the Databricks
job config, single-node g5 cluster payload generation, runner script emission,
and CLI entry point. The legacy
`restaurant_kv_serving.databricks_vllm_smoke_job` module remains as a
compatibility wrapper.

## Review

GPT-5.5 reviewed the ownership inversion and found no behavioral regressions.
It recommended two additional guardrails:

- restoration coverage when a projected legacy monkeypatch raises during
  `main()`;
- exact legacy `__all__` coverage for the previous no-`__all__` star-import
  surface.

Both guardrails were added to `tests/test_databricks_vllm_smoke_job.py`.

A follow-up GPT-5.5 review of the same reverse-wrapper pattern found that
document-namespace monkeypatches, including private helper monkeypatches, could
leak into legacy calls. The legacy wrapper now executes cloned document
functions against an isolated globals dictionary instead of mutating the
document module. Regression tests cover public hook isolation, private helper
isolation, and actual legacy star import behavior.

## Verification

- `poetry run pytest tests/test_databricks_vllm_smoke_job.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/databricks_vllm_smoke_job.py src/restaurant_kv_serving/databricks_vllm_smoke_job.py tests/test_databricks_vllm_smoke_job.py tests/test_public_package.py`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `poetry build`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- GPT-5.5 review and re-review after resolving legacy/document monkeypatch
  isolation findings
