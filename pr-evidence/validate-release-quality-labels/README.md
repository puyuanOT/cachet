# Validate Release Quality Labels

## Scope

- Recompute V1 measurement `exact_match` and `answer_found` labels from `output_text` and `expected_answer` during release evidence validation.
- Preserve existing field-level validation for malformed `output_text`, `expected_answer`, and non-boolean quality labels.
- Use the benchmark module's quality helpers so release validation matches benchmark-runner semantics.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_incorrect_measurement_quality_labels tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_malformed_measurement_quality_flags tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
