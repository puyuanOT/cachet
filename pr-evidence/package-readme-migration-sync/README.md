# Package README Migration Sync

This evidence covers the PR that refreshes package-level README documentation
after the document namespace ownership migration.

The document package README now describes `document_kv_cache` as the canonical
implementation namespace. The legacy restaurant package README now describes
`restaurant_kv_serving` as migration shims that forward to document-owned
implementations, with `scheduler.py` called out as the older admission-helper
shim.

Verification:

- `poetry run pytest tests/test_project_governance.py -q`
- `poetry run pytest`
- `poetry check`
- `poetry build`
- `git diff --check`
- repository secret scan
- GPT-5.5 focused review with findings resolved

