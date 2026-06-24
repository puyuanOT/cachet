# Document Benchmark Plan Ownership

This evidence covers the PR that moves benchmark plan ownership from
`restaurant_kv_serving.benchmark_plan` to `document_kv_cache.benchmark_plan`.

The legacy restaurant module is now a compatibility facade. It delegates to the
document module while preserving legacy monkeypatch behavior for tests and
downstream callers that still import the old path.

Verification:

- `poetry run pytest`
- `poetry check`
- `poetry build`
- `git diff --check`
- repository secret scan
- GPT-5.5 focused review with findings resolved

