# Validate Release Comparison Aggregates

## Scope

- Recompute V1 comparison `ttft_speedup` and `time_to_completion_speedup` from baseline/cache report-row p50 latency values.
- Reject finite comparison speedups when report-row p50 values are valid but non-positive, matching producer `_speedup` semantics.
- Recompute comparison `exact_match_delta` and `answer_found_delta` from report-row quality rates when both source rates are present.
- Keep optional quality-rate compatibility: omitted report-row rates are not forced solely to validate a comparison field.
- Tighten release and release-bundle fixtures so synthetic release-ready artifacts are internally consistent.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_comparison_report_row_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_comparison_speedup_for_zero_report_p50 tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `git diff --check`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
