# Validate Release Quality Rates

## Scope

- Reconcile V1 release report `exact_match_rate` and `answer_found_rate` fields against raw successful measurements when those report fields are present.
- Preserve existing compatibility for report rows that omit one of the optional quality-rate fields.
- Keep quality-rate validation scoped to successful measurements, matching the existing token and throughput aggregate checks.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_report_row_quality_rate_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_report_row_aggregate_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
- `poetry run pytest -q`
