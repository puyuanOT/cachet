# Document Benchmark Ownership

This PR-evidence sidecar covers the refactor slice that moves the V1 benchmark
contract implementation from the legacy restaurant package into
`document_kv_cache.benchmarks`.

The slice makes `document_kv_cache.benchmarks` the implementation owner for V1
dataset specs, prompt partitioning, measurement summaries, quality helpers, and
baseline-vs-cache comparisons. `restaurant_kv_serving.benchmarks` remains as a
compatibility wrapper for existing jobs and tests.

## Review

GPT-5.5 found one compatibility issue: the legacy benchmark module used to
expose `SourceDocument` at module scope because it imported it directly. The
initial wrapper omitted that name, which could break legacy direct or star
imports.

The finding was fixed by re-exporting `SourceDocument` only from the legacy
wrapper. The document benchmark module keeps its curated benchmark `__all__`
surface, while the legacy wrapper preserves the broader compatibility surface.
The final GPT-5.5 review found no remaining findings.

## Verification

- `poetry run pytest tests/test_benchmarks.py tests/test_public_package.py tests/test_benchmark_runner.py tests/test_dataset_prep.py tests/test_release_evidence.py -q`
- `poetry run pytest tests/test_public_package.py tests/test_benchmarks.py tests/test_project_governance.py::test_legacy_restaurant_imports_in_tests_are_explicitly_scoped -q`
- `python -m py_compile src/document_kv_cache/benchmarks.py src/restaurant_kv_serving/benchmarks.py tests/test_public_package.py tests/test_project_governance.py`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry build`
- `PYTHONPATH=src python -m document_kv_cache.pr_evidence --validate-directory pr-evidence --output-json /tmp/pr-evidence-document-benchmark-ownership-validation.json`
