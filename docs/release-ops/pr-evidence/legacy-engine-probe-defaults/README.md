# Legacy Engine Probe Defaults Evidence

This evidence covers the compatibility refactor that isolates
`restaurant_kv_serving.engine_probe` class bases from pre-import public
`document_kv_cache.engine_probe` mutations.

The legacy wrapper now preserves clean-import subclass relationships for
`EngineKVProbeConfig`, `EngineKVProbeFactoryContext`, and
`EngineKVProbeFactoryResult`, while falling back to source-loaded document
classes when the public classes were replaced or mutated before legacy import.
It also keeps the factory-result type check stable when the public result class
is mutated after a clean legacy import.

Verification:

- `poetry run pytest tests/test_engine_probe.py`
- `poetry run pytest tests/test_engine_probe.py tests/test_engine_adapters.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry build`
- GPT-5.5 review with one finding resolved and final approval
