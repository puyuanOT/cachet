# Storage URI Handoff Writer

This PR evidence covers the engine adapter handoff writer change that resolves
storage-style output paths through the shared `local_path` helper.

## Verification

- `poetry run pytest tests/test_engine_adapters.py -q`
- `poetry run pytest tests/test_engine_adapters.py tests/test_project_governance.py tests/test_public_package.py -q`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check origin/main...HEAD`

## Review

GPT-5.5 approved the committed diff with no blocking findings. The reviewer ran
adapter, storage, engine-probe, public-package, and diff-check verification and
noted only that live Databricks `/dbfs` or `/Volumes` mounts were not exercised.
