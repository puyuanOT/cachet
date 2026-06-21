# Validate Release Report Aggregates

## Summary

- Extended strict V1 release evidence reconciliation from report-row counts to report-row aggregate metrics.
- Validated `prompt_tokens_mean`, `completion_tokens_mean`, and `output_tokens_per_second` against successful raw measurement rows.
- Added regression coverage for tampered report-row aggregate metrics, zero-denominator throughput, and repeated-measurement behavior.

## Why

The raw measurement rows are the source of truth for V1 benchmark evidence. Release report rows summarize latency, quality, token volume, and throughput; this PR prevents those summary aggregates from drifting away from the trace they claim to summarize.

## Refactor Scope

The existing report-row count reconciliation helper was expanded into a single aggregate reconciliation helper. A small internal aggregate dataclass keeps request/error counts, token sums, and throughput inputs in one place without changing public release evidence records.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_report_row_aggregate_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_throughput_when_measurements_have_zero_total_time tests/test_release_evidence.py::test_evaluate_release_evidence_allows_repeated_raw_measurements tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
- `git diff --check`
