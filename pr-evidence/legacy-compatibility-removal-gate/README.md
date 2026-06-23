# Legacy Compatibility Removal Gate PR Evidence

This folder contains PR evidence for documenting and enforcing the
migration-only `restaurant_kv_serving` removal gate.

The PR keeps Cachet on one repository and one distribution package while making
legacy facade removal depend on machine-checkable downstream migration
evidence. It adds the `document_kv.legacy_compatibility_migration.v1` validator,
the optional `legacy_migration_evidence` release-bundle role, and a production
source governance guard that rejects new `restaurant_kv_serving` dependencies
outside the legacy compatibility package.

Verification:

- `poetry run pytest tests/test_legacy_compatibility.py -q`
- `poetry run pytest tests/test_project_governance.py -q`
- `poetry run pytest tests/test_release_bundle.py -q`
- `poetry run pytest tests/test_public_package.py -q`
- `poetry run pytest -q`
- GPT-5.5 focused review returned two findings; both were fixed and the same
  reviewer approved the follow-up.
