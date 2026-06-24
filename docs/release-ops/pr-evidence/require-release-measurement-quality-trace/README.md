# Require Release Measurement Quality Trace

## Summary

- Tightened strict V1 release-evidence validation for raw benchmark measurement quality fields.
- Release measurements now require a non-empty `expected_answer`.
- Release measurements now require boolean `exact_match` and `answer_found` values.
- Updated release evidence and release bundle fixtures to carry the full raw quality trace.

## Why

V1 release artifacts need enough raw measurement data to audit quality claims. Cachet-generated benchmark records already include `expected_answer`, `output_text`, `exact_match`, and `answer_found`; strict release validation now enforces that externally assembled records do not lose those fields.

## Refactor Evidence

This is a narrow release-validation hardening change in the existing V1 evidence path. It does not change public APIs or generated benchmark schemas; it enforces fields already emitted by the benchmark writer.

## Verification

```text
poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_malformed_measurement_quality_flags tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_malformed_measurement_trace_fields tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_stub_measurement_rows tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q
4 passed in 0.11s

poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q
109 passed in 1.56s

poetry run pytest -q
1235 passed in 9.15s

poetry check
All set!

poetry run python -m compileall -q src tests
passed

git diff --check
passed
```
