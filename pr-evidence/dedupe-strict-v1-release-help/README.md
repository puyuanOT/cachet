# PR Evidence: dedupe-strict-v1-release-help

## What changed

- Centralized the strict V1 release-bundle CLI help text in `document_kv_cache.release_bundle`.
- Reused the shared help constant from the legacy `restaurant_kv_serving.release_bundle` facade.
- Added a regression test that checks both parser actions use the public shared value.

## Verification

- `poetry run pytest tests/test_release_bundle.py -q`
- `poetry run pytest tests/test_release_bundle.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry run python -m compileall -q src tests`

## Review

GPT-5.5 reviewer Beauvoir the 3rd approved with no findings.
