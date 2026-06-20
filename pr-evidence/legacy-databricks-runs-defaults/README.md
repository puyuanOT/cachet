# Legacy Databricks Runs Defaults Evidence

This evidence covers the compatibility refactor that isolates
`restaurant_kv_serving.databricks_runs` from pre-import monkeypatches in
`document_kv_cache.databricks_runs`.

The legacy wrapper now loads source-default document module state for isolated
function cloning and falls back to source-default bases only when the public
document classes have been replaced before legacy import. Clean imports still
preserve the expected public protocol identity and legacy subclass relationship.

Verification:

- `poetry run pytest tests/test_databricks_runs.py`
- `poetry run pytest tests/test_databricks_runs.py tests/test_public_package.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry build`
- GPT-5.5 review with one finding resolved and final approval
