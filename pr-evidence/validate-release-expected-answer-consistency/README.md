# Validate Release Expected Answer Consistency

## Scope

- Require all valid V1 measurements for the same `(dataset, example_id)` identity to use the same `expected_answer`.
- Preserve existing malformed-field validation; invalid or empty expected answers are still reported by the measurement field validator.
- Keep repeated raw measurement support unchanged.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_inconsistent_measurement_expected_answers tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_unpaired_measurement_examples tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
