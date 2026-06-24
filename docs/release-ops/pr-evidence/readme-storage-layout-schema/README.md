# README Storage Layout Schema Evidence

This PR-evidence sidecar covers the documentation contract slice that aligns the
README manifest table with the persisted `storage_layout` field.

The slice adds `storage_layout` to the manifest schema example in the root
README and adds a governance test that inspects the actual fenced manifest table
block, so the field cannot drift into unrelated prose while disappearing from
the schema.

## Review

GPT-5.5 first found that the governance assertion was too loose because it only
looked for `storage_layout` somewhere in the Logical Model section. The test now
extracts the manifest table fenced block containing `Manifest table:` and checks
that `storage_layout` appears immediately after `layout_version`.

The final GPT-5.5 pass found no remaining issues.

## Verification

- `poetry run pytest tests/test_project_governance.py::test_readme_manifest_schema_mentions_storage_layout tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry build`
