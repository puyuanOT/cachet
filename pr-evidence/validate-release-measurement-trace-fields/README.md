# Validate Release Measurement Trace Fields

## Summary

- Tightened strict V1 release-evidence validation for raw benchmark measurements.
- Release records now require every measurement to include a non-empty `example_id` and a string `output_text`.
- Updated release evidence and release bundle fixtures to include raw output text.

## Why

V1 release artifacts need enough raw measurement detail to audit benchmark traceability and quality. Records written by Cachet already include these fields, but externally assembled records could previously pass release validation without an example id or output text.

## Refactor Evidence

This is a small release-validation hardening change in the existing validator path. It does not change public APIs or generated benchmark schemas; it makes strict release checks enforce fields already produced by the benchmark writer.

## Verification

```text
poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_malformed_measurement_trace_fields tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_stub_measurement_rows tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q
3 passed in 0.12s

poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q
109 passed in 1.57s

poetry run pytest -q
1235 passed in 8.98s

poetry check
All set!

poetry run python -m compileall -q src tests
passed

git diff --check
passed
```
