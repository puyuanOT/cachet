# Require Release Quality Rates

## Scope

- Require every strict V1 report row to include both `exact_match_rate` and
  `answer_found_rate`.
- Keep existing rate validation and aggregate cross-checks, so comparison
  quality deltas are backed by row-level quality data.
- Update release-ready test fixtures to emit both quality rates.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_requires_both_report_quality_rates tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_malformed_report_quality_rates tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_report_row_quality_rate_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
