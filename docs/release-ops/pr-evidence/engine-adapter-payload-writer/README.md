# Engine Adapter Payload Writer

This PR evidence covers the public helper for writing validated engine adapter
payload bytes to local, DBFS, or Unity Catalog storage before emitting a vLLM or
SGLang handoff record.

## Verification

- `poetry run pytest tests/test_engine_probe.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check origin/main...HEAD`

## Review

GPT-5.5 approved the committed diff with no blocking findings. The reviewer ran
focused engine-probe, public-package, governance, engine-adapter, storage, and
diff-check verification and reported no behavior, API, path-safety, legacy
wrapper, typing, or scope issues.
