# Document KVPack Ownership

This PR-evidence sidecar covers the refactor slice that moves packed KV shard
writing into `document_kv_cache.kvpack`.

The slice makes the document package the implementation owner for `PackChunk`,
`LocalRangeReader`, and `write_kvpack`. The legacy
`restaurant_kv_serving.kvpack` module remains as a compatibility wrapper. It
keeps `PackChunk` aliased to the document-owned dataclass while preserving old
legacy behavior where `LocalRangeReader` aliases the legacy `DiskRangeReader`
and `write_kvpack` honors `restaurant_kv_serving.kvpack.local_path` overrides.

## Review

GPT-5.5 reviewed the KVPack ownership inversion before merge. The initial
review found two legacy compatibility regressions: legacy `LocalRangeReader`
no longer aliased the legacy `DiskRangeReader`, and legacy `write_kvpack` no
longer honored monkeypatches of the legacy module's `local_path`. Both were
fixed by factoring the document writer through a private path-resolver helper
and using a legacy wrapper for writes. Re-review reported no findings and
approved the diff.

## Verification

- `python -m py_compile src/document_kv_cache/kvpack.py src/restaurant_kv_serving/kvpack.py tests/test_kvpack.py tests/test_public_package.py tests/test_project_governance.py`
- `poetry run pytest tests/test_kvpack.py tests/test_cache.py tests/test_storage.py tests/test_workflow.py tests/test_engine.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest tests/test_kvpack.py tests/test_cache.py tests/test_storage.py tests/test_public_package.py tests/test_project_governance.py -q`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry build`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- `PYTHONPATH=src python - <<'PY' ... PY` import sanity check for document
  and legacy KVPack aliases
- GPT-5.5 review before merge
