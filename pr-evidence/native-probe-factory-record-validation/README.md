# Native Probe Factory Record Validation Evidence

This sidecar records verification for exposing public validators for `document_kv.native_probe_factories.v1` diagnostics and reusing them from release bundle validation.

## Verification

- `poetry run pytest tests/test_native_probe_factories.py tests/test_release_bundle.py tests/test_public_package.py tests/test_project_governance.py -q` -> 97 passed.
- `git diff --check` -> clean.
- `poetry run pytest -q` -> 946 passed.
- `poetry build` -> succeeded.
- `poetry check && poetry install --dry-run` -> succeeded.

## Review

GPT-5.5 high review approved with no findings and confirmed release-bundle behavior parity, export compatibility, and no circular import risk.
