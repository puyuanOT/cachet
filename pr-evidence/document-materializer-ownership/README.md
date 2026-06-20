# Document Materializer Ownership

This PR-evidence sidecar covers the refactor slice that moves KV cache
materialization into `document_kv_cache.materializer`.

The slice makes the document package the implementation owner for contiguous
and segmented payload materialization, segment byte-offset validation,
materialization timing, and cache-tier normalization. The legacy
`restaurant_kv_serving.materializer` module remains as a compatibility wrapper
that aliases the document-owned materializer objects and keeps
`normalize_segment_tiers` available for legacy engine imports.

## Review

GPT-5.5 reviewed the materializer ownership inversion before merge. The review
checked offset and byte-length validation, segmented materialization behavior,
cache-tier normalization, legacy alias compatibility, star imports, root
exports, and circular-import risk.

## Verification

- `poetry run pytest tests/test_planner_materializer.py tests/test_scheduler.py tests/test_engine.py tests/test_workflow.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/materializer.py src/restaurant_kv_serving/materializer.py src/document_kv_cache/__init__.py tests/test_planner_materializer.py tests/test_public_package.py tests/test_project_governance.py`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry build`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- GPT-5.5 review before merge
