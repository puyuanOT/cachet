# Document vLLM Smoke Ownership

This PR-evidence sidecar covers the refactor slice that moves the
Databricks-friendly vLLM smoke implementation into
`document_kv_cache.vllm_smoke`.

The slice makes the document package the implementation owner for the isolated
vLLM environment setup, Qwen3 4B Instruct server launch arguments, smoke
dataset generation, benchmark-runner invocation, and CLI parsing. The legacy
`restaurant_kv_serving.vllm_smoke` module remains as a compatibility wrapper.

## Review

GPT-5.5 initially found three compatibility issues:

- legacy `python -m restaurant_kv_serving.vllm_smoke` no longer executed
  `main()`;
- broad legacy helper exports such as `create_venv` and `run` were not
  monkeypatch-compatible when called directly;
- the Databricks smoke job still imported smoke defaults through the legacy
  wrapper.

Those issues were fixed by restoring the legacy `__main__` block, adding
legacy helper wrappers that project only genuinely monkeypatched legacy globals
into the document implementation, and importing smoke defaults from
`document_kv_cache.vllm_smoke` in the Databricks smoke job.

GPT-5.5 then found a lower-severity typing compatibility issue because the
helper wrappers used generic `*args, **kwargs` signatures in a `py.typed`
package. The wrappers now preserve concrete signatures matching the document
implementation. Final re-review found no issues.

## Verification

- `poetry run pytest tests/test_vllm_smoke.py tests/test_databricks_vllm_smoke_job.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/vllm_smoke.py src/restaurant_kv_serving/vllm_smoke.py src/restaurant_kv_serving/databricks_vllm_smoke_job.py tests/test_vllm_smoke.py tests/test_databricks_vllm_smoke_job.py tests/test_public_package.py`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- GPT-5.5 re-review after resolving legacy execution, monkeypatch, import-boundary, and signature findings
