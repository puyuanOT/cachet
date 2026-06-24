# Document Databricks Storage Benchmark Job Ownership

This PR-evidence sidecar covers the refactor slice that moves the Databricks
`runs/submit` payload helper for storage-reader benchmarks into
`document_kv_cache.databricks_storage_benchmark_job`.

The slice makes the document package the implementation owner for the storage
benchmark job config, single-node g5 cluster payload generation, runner script
emission, and CLI entry point. The legacy
`restaurant_kv_serving.databricks_storage_benchmark_job` module remains as a
compatibility wrapper.

## Review

GPT-5.5 reviewed the ownership inversion and found compatibility issues in the
initial reverse-wrapper implementation. Those issues were fixed by:

- running legacy calls through cloned document functions with an isolated
  globals dictionary rather than mutating document module globals;
- preserving legacy validator and private cluster-helper monkeypatch behavior;
- using an honest legacy config subclass for direct legacy construction,
  pickle round-trips, and slotted layout;
- adding actual legacy star-import coverage and explicit
  `document_kv_cache.storage.is_real_uc_volume_root` coverage.

Final GPT-5.5 re-review found no remaining issues.

## Verification

- `poetry run pytest tests/test_databricks_storage_benchmark_job.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/databricks_storage_benchmark_job.py src/restaurant_kv_serving/databricks_storage_benchmark_job.py src/document_kv_cache/storage.py tests/test_databricks_storage_benchmark_job.py tests/test_public_package.py`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `poetry build`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- GPT-5.5 review and re-review after resolving all compatibility findings
