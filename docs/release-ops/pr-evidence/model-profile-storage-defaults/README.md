# Model Profile Storage Defaults

This PR lets portable model profiles own their default K/V storage layout, so
future model bundles can derive shared, separate, or interleaved KV handoffs
without every caller passing the same override.

## Verification

- `poetry run pytest tests/test_model_profiles.py -q`
- `poetry run pytest tests/test_model_profiles.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`
- `git diff --check`

## Review

GPT-5.5 approved the committed implementation diff with no blocking findings.
