# Document Benchmark Runner Ownership

This PR-evidence sidecar covers the refactor slice that moves V1 benchmark
execution from the legacy restaurant package into
`document_kv_cache.benchmark_runner`.

The slice makes `document_kv_cache.benchmark_runner` the implementation owner
for canonical JSONL loading, baseline/cache-arm execution, OpenAI-compatible
benchmark configuration, result serialization, and the benchmark-runner CLI.
`restaurant_kv_serving.benchmark_runner` remains as a compatibility wrapper for
existing jobs and tests.

## Review

GPT-5.5 review found no issues. The review checked the ownership inversion,
legacy wrapper export surface, `_openai_compatible_engine` compatibility, CLI
help under both module paths, import-cycle behavior around
`document_kv_cache.openai_compatible`, and the legacy monkeypatch bridge into
document-owned `main()`.

The document module keeps a curated `__all__` for the public benchmark-runner
API. The legacy wrapper intentionally preserves the broader historical
module-level import surface and direct `_openai_compatible_engine` access.

## Verification

- `poetry run pytest tests/test_benchmark_runner.py tests/test_public_package.py tests/test_openai_compatible.py tests/test_live_server.py -q`
- `python -m py_compile src/document_kv_cache/benchmark_runner.py src/restaurant_kv_serving/benchmark_runner.py tests/test_benchmark_runner.py tests/test_public_package.py`
- `poetry run pytest tests/test_benchmark_runner.py tests/test_public_package.py tests/test_benchmark_plan.py tests/test_benchmark_plan_executor.py tests/test_openai_compatible.py tests/test_live_server.py tests/test_vllm_smoke.py tests/test_release_evidence.py tests/test_project_governance.py::test_legacy_restaurant_imports_in_tests_are_explicitly_scoped -q`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry build`
- `PYTHONPATH=src python -m document_kv_cache.pr_evidence --validate-directory pr-evidence --output-json /tmp/pr-evidence-document-benchmark-runner-ownership-validation.json`
