# V1 Databricks Purpose Constant

## Summary

The V1 benchmark Databricks job now exposes a dedicated
`DEFAULT_DATABRICKS_PURPOSE` constant. The value remains
`document-kv-v1-benchmark`, but release validation no longer has to treat the
run name as the purpose source.

This keeps the V1 benchmark job aligned with the storage benchmark, vLLM smoke,
and engine-probe helpers, which already expose separate run-name, task-key, and
purpose constants.

## Verification

```bash
pytest tests/test_databricks_job.py tests/test_release_bundle.py tests/test_public_package.py
```
