# Document Models Ownership

This PR-evidence sidecar covers the refactor slice that moves cache keys,
chunk references, request objects, and materialization plan models into
`document_kv_cache.models`.

The slice makes the document package the implementation owner for
`KVCacheKey`, `ChunkRef`, `DocumentKVRequest`, `PlanSegment`,
`MaterializationPlan`, document chunk enums, cache generation methods, and
chunk-type helper functions. The legacy `restaurant_kv_serving.models` module
remains as a compatibility wrapper that aliases the document-owned objects and
keeps old restaurant-specific compatibility names such as `ChunkType`,
`ChunkId`, and `RestaurantKVRequest` available.

## Review

GPT-5.5 reviewed the models ownership inversion before merge. The initial
review found that the legacy wrapper had dropped the old `ChunkId` module
attribute. The wrapper and compatibility test were patched, and the reviewer
approved the updated diff with no remaining findings.

## Verification

- `python -m py_compile src/document_kv_cache/models.py src/restaurant_kv_serving/models.py src/document_kv_cache/__init__.py tests/test_planner_materializer.py tests/test_public_package.py tests/test_project_governance.py`
- `poetry run pytest tests/test_planner_materializer.py tests/test_cache.py tests/test_storage.py tests/test_kvpack.py tests/test_scheduler.py tests/test_engine.py tests/test_workflow.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/models.py src/restaurant_kv_serving/models.py tests/test_planner_materializer.py`
- `poetry run pytest tests/test_planner_materializer.py tests/test_public_package.py tests/test_project_governance.py -q`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry build`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- `PYTHONPATH=src python - <<'PY' ... PY` import sanity check for document
  and legacy model aliases
- GPT-5.5 review before merge
