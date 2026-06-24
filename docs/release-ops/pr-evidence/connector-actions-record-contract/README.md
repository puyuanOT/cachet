# Connector Actions Record Contract PR Evidence

## What Changed

- Added a JSON-compatible `document_kv.engine_kv_connector_actions.v1` record
  contract for native vLLM/SGLang connector reserve/copy/bind/release action
  descriptors.
- Exposed the record serializer, parser, validator, and schema constants through
  both the document package and the legacy compatibility package.
- Documented the out-of-process native adapter handoff path and added regression
  tests for record round trips, stale field rejection, required `source_byte_end`,
  and legacy star exports.

## Verification

- `poetry run pytest tests/test_engine_adapters.py -k 'connector_actions_record or document_module_owns_public_api' tests/test_public_package.py -k 'core_api or star_exports_are_document_first or curated_star_import_surfaces' -q`
- `poetry run pytest tests/test_engine_adapters.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry build`

## GPT-5.5 Review

The GPT-5.5 reviewer initially requested two fixes: expose the new connector
actions symbols through the legacy package's explicit `__all__`, and require
`source_byte_end` when parsing v1 copy descriptors. Both were patched and covered
by regression tests. The reviewer re-checked the branch and approved it.
