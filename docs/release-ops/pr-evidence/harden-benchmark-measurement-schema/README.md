# Harden Benchmark Measurement Schema

This PR tightens producer-side validation for benchmark measurements.

`InferenceMeasurement` now validates the traceability and serialization fields that are later written into V1 benchmark artifacts:

- `example_id` and `arm_id` must be non-empty strings.
- `output_text` must be a string, with empty strings still allowed for error paths or blank model generations.
- Optional `expected_answer` and `error` must be non-empty when provided.
- `metadata` is normalized through the existing string-mapping validator.

The change keeps valid benchmark outputs compatible while rejecting malformed measurement objects before they reach benchmark summaries or release evidence.
