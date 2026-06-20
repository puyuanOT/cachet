# Document Databricks Runs Ownership

This PR-evidence sidecar covers the refactor slice that moves the Databricks
Jobs `runs/submit` and `runs/get` helper implementation into
`document_kv_cache.databricks_runs`.

The slice makes the document package the implementation owner for Databricks
workspace config loading, run submission, run status fetching, sanitized summary
records, and the CLI entry point. The legacy
`restaurant_kv_serving.databricks_runs` module remains as a compatibility
wrapper.

## Review

GPT-5.5 reviewed the ownership inversion and found one credential-hygiene issue:
Databricks HTTP error bodies could echo bearer credentials, and the CLI would
write the resulting exception message into `--output-json`.

The implementation now redacts the configured Databricks token and generic
`Bearer ...` credential echoes before raising HTTP errors. Regression tests cover
both direct API calls and the CLI JSON-output path. GPT-5.5 re-reviewed the fix
and approved the branch.

## Verification

- `poetry run pytest tests/test_databricks_runs.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/databricks_runs.py src/restaurant_kv_serving/databricks_runs.py src/document_kv_cache/__init__.py tests/test_databricks_runs.py tests/test_public_package.py`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry build`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- GPT-5.5 review and re-review after resolving the credential-redaction finding
