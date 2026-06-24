# Strict V1 Databricks Identity

## Summary

Strict V1 release bundles now bind each required Databricks purpose to the
canonical release run name and task key:

- `document-kv-v1-benchmark` -> `document-kv-v1-benchmark` / `document_kv_v1_benchmark`
- `document-kv-storage-benchmark` -> `document-kv-storage-benchmark` / `document_kv_storage_benchmark`
- `document-kv-engine-probe` -> `document-kv-engine-probe` / `document_kv_engine_probe`

This prevents a stale or custom Databricks run-status sidecar from satisfying
strict V1 release evidence merely by retaining the expected purpose tag.

## Verification

```bash
pytest tests/test_release_bundle.py::test_build_release_bundle_strict_v1_rejects_databricks_purpose_identity_mismatch
```
