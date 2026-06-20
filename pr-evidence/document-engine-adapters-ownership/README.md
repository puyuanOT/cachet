# Document Engine Adapters Ownership

This evidence covers the PR that moves the external serving-engine adapter
contract implementation into `document_kv_cache.engine_adapters`.

The document module now owns the vLLM/SGLang adapter request, injection-plan,
connector-action, probe-record, and payload-source helpers. The legacy
`restaurant_kv_serving.engine_adapters` module remains a compatibility facade
and preserves the old broad non-private import surface for existing callers.

Verification:

- `poetry run pytest tests/test_engine_adapters.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest`
- `poetry check`
- `poetry build`
- `git diff --check`
- repository secret scan
- GPT-5.5 focused review with findings resolved

