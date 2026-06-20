# Document Databricks Benchmark Job Ownership

This PR-evidence sidecar covers the refactor slice that moves the Databricks
`runs/submit` payload helper for V1 benchmark-plan execution into
`document_kv_cache.databricks_job`.

The slice makes the document package the implementation owner for the reusable
single-node AWS g5 cluster config, V1 benchmark job config, runner script
emission, and CLI entry point. The legacy `restaurant_kv_serving.databricks_job`
module remains as a compatibility wrapper.

## Review

GPT-5.5 reviewed the ownership inversion and found one document-root export
leak: `document_kv_cache` root symbols still resolved Databricks job classes and
functions through the legacy root. The root export map now redirects Databricks
job symbols to `document_kv_cache.databricks_job`, tests cover document-root
imports/star imports and direct legacy private-hook construction, and re-review
approved the branch with no remaining findings.

## Verification

- `poetry run pytest tests/test_databricks_job.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/__init__.py src/document_kv_cache/databricks_job.py src/restaurant_kv_serving/databricks_job.py tests/test_databricks_job.py tests/test_public_package.py`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- `poetry check`
- `poetry build`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- GPT-5.5 review and re-review after resolving the document-root export finding
