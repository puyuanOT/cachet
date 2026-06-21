# Validate V1 benchmark identities

## Summary

- Added duplicate identity diagnostics to `V1BenchmarkEvidence`.
- Detect duplicate required datasets, duplicate `(dataset, arm_id)` report rows, and duplicate target-arm comparisons.
- Serialize the duplicate evidence fields in benchmark runner output.
- Make release evidence reject non-empty V1 duplicate fields.
- Reject duplicate dataset IDs in `BenchmarkSuite`.

## Why

The V1 release gate requires one complete benchmark summary for Biography, HotpotQA, MusiQue, and NIAH across baseline prefill and document KV-cache arms. Duplicate report rows or comparisons could previously be silently overwritten by dictionary construction, allowing ambiguous benchmark evidence to look release-ready.

## Refactor Evidence

- Applied the Refactor skill.
- Extracted small identity helpers rather than duplicating map/duplicate logic inline.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_duplicate_v1_evidence_identities tests/test_release_evidence.py::test_evaluate_release_evidence_allows_legacy_v1_evidence_without_duplicate_identity_fields tests/test_benchmarks.py::test_evaluate_v1_benchmark_evidence_reports_duplicate_identities -q`
  - `3 passed`
- `poetry run pytest tests/test_benchmarks.py::test_benchmark_suite_validates_identity_examples_and_datasets tests/test_benchmarks.py::test_evaluate_v1_benchmark_evidence_accepts_complete_v1_result tests/test_benchmarks.py::test_evaluate_v1_benchmark_evidence_reports_duplicate_identities tests/test_benchmark_runner.py::test_benchmark_run_result_to_record_serializes_latency_quality_and_comparison tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_duplicate_v1_evidence_identities -q`
  - `5 passed`
- `poetry run pytest tests/test_benchmarks.py tests/test_benchmark_runner.py tests/test_release_evidence.py tests/test_release_bundle.py -q`
  - `194 passed`
- `poetry run pytest -q`
  - `1233 passed`
- `poetry check`
  - `All set!`
- `poetry run python -m compileall -q src tests`
- `git diff --check`

## Review

- GPT-5.5 subagent found one compatibility issue: the new duplicate evidence fields were treated as required by release evidence.
- Fixed by treating missing duplicate fields as empty for legacy benchmark records.
- Added a regression for old-style V1 evidence records without duplicate identity fields.
