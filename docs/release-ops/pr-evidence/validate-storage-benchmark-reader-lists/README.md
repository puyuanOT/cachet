# Validate storage benchmark reader lists

## Summary

- Added a shared storage benchmark reader-list validator.
- `StorageBenchmarkConfig` now normalizes supported reader sequences to tuples.
- Duplicate, scalar, empty, non-string, and unsupported reader ids are rejected before benchmark execution or evidence evaluation.

## Verification

- `poetry run pytest tests/test_storage_benchmark.py::test_evaluate_storage_benchmark_evidence_validates_required_readers tests/test_storage_benchmark.py::test_storage_benchmark_config_validates_inputs -q`
  - `2 passed`
- `poetry run pytest tests/test_storage_benchmark.py -q`
  - `22 passed`
- `poetry run pytest -q`
  - `1228 passed`
- `poetry check`
  - `All set!`
- `poetry run python -m compileall -q src tests`
- `git diff --check`

## Review

- GPT-5.5 subagent review approved with no blocking issues.
- Reviewer spot-check command:
  - `poetry run pytest tests/test_storage_benchmark.py tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_duplicate_or_missing_storage_readers tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_invalid_storage_result_readers -q`
  - `28 passed`
