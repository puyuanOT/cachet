# Document Planner Ownership

This PR-evidence sidecar covers the refactor slice that moves runtime
materialization planning into `document_kv_cache.planner`.

The slice makes the document package the implementation owner for
`CacheRequest` and `CachePlanner`. The legacy `restaurant_kv_serving.planner`
module remains as a compatibility wrapper that aliases the document-owned
planner objects and keeps old broad module imports available for existing
integrations.

## Review

GPT-5.5 reviewed the planner ownership inversion before merge. The review
checked task-prefix handling, static/content chunk ordering, token and byte
cursors, selected document ids, legacy restaurant request behavior, import
cycles after the models and manifest moves, document and legacy star-import
surfaces, root export behavior, and governance allowlist scope. It reported no
findings and approved the diff.

## Verification

- `python -m py_compile src/document_kv_cache/planner.py src/restaurant_kv_serving/planner.py src/document_kv_cache/__init__.py tests/test_planner_materializer.py tests/test_public_package.py tests/test_project_governance.py`
- `poetry run pytest tests/test_planner_materializer.py tests/test_engine.py tests/test_workflow.py tests/test_engine_adapters.py tests/test_engine_probe.py tests/test_public_package.py tests/test_project_governance.py -q`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry build`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- `PYTHONPATH=src python - <<'PY' ... PY` import sanity check for document
  and legacy planner aliases
- GPT-5.5 review before merge
