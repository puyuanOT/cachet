# Validate Release Latency Summaries

## Scope

- Cross-check strict V1 report-row latency `mean`, `p50`, and `p95` values
  against successful raw measurements.
- Reuse the same sorted linear interpolation percentile semantics as benchmark
  report generation.
- Preserve existing latency shape, count, and comparison validation.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_latency_summary_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_latency_count_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_report_row_aggregate_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
