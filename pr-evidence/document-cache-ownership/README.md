# Document Cache Ownership

This PR-evidence sidecar covers the refactor slice that moves the tiered byte
cache implementation into `document_kv_cache.cache`.

The slice makes the document package the implementation owner for cache tier
labels, cache hit/miss stats, byte-accounted CPU LRU behavior, local-disk cache
reuse, local cache budget eviction, and checksum validation of cached payloads.
The legacy `restaurant_kv_serving.cache` module remains as a compatibility
wrapper that aliases the document-owned cache objects.

## Review

GPT-5.5 reviewed the cache ownership inversion before merge. The review checked
tiered cache behavior, alias-based legacy compatibility, root exports, star
imports, and circular-import risk.

## Verification

- `poetry run pytest tests/test_cache.py tests/test_planner_materializer.py tests/test_engine.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/cache.py src/restaurant_kv_serving/cache.py src/document_kv_cache/__init__.py tests/test_cache.py tests/test_public_package.py tests/test_project_governance.py`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry build`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- GPT-5.5 review before merge
