# Validate Release Report Counts

## Summary

- Added strict V1 release evidence validation that compares each report row's `requests` and `errors` fields against the raw measurement rows for the same dataset and benchmark arm.
- Kept repeated raw measurements valid when the report row request count agrees with the measurement trace.
- Added regression coverage for mismatched report-row counts.

## Why

V1 release bundles use report rows for latency and quality summaries, but the raw measurements are the traceable source of truth. This PR prevents a release artifact from claiming request or error counts that are not supported by the measurement trace.

## Refactor Scope

The change stays inside the existing release evidence validator. It adds a small count-consistency helper and a shared dataset/arm predicate so existing measurement and report-row validation responsibilities remain separate.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_report_row_count_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_allows_repeated_raw_measurements tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `poetry run pytest tests/test_release_evidence.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
- `git diff --check`
