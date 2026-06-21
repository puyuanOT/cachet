# Validate Release Measurement Pairs

## Scope

- Require every valid V1 `(dataset, example_id)` measurement identity to include both `baseline_prefill` and `document_kv_cache` arms.
- Preserve support for repeated raw measurements; report-row aggregate validation still reconciles request and latency counts.
- Keep malformed measurement fields handled by the existing field-level validators.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_unpaired_measurement_examples tests/test_release_evidence.py::test_evaluate_release_evidence_allows_repeated_raw_measurements tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
