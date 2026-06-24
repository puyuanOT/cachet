# Validate Release Latency Counts

## Summary

- Required strict V1 report-row latency summaries to include `count` and `mean` in addition to `p50` and `p95`.
- Added validation that each report-row `ttft.count` and `time_to_completion.count` matches the successful raw measurements for the same dataset and benchmark arm.
- Updated release evidence and bundle fixtures to mirror the benchmark runner's serialized latency summary shape.

## Why

The benchmark runner writes complete latency summaries, and release evidence should preserve that traceability. This PR prevents a release artifact from reporting latency percentiles over a different number of successful requests than the raw measurement trace proves.

## Refactor Scope

The generic latency helper remains shared with storage benchmarks; `count` and `mean` are required only for V1 report-row latency summaries so existing storage p50/p95 validation stays unchanged.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_latency_count_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_report_row_count_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
- `git diff --check`
