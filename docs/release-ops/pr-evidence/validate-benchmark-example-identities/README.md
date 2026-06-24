# Validate Benchmark Example Identities

## Summary

- Added `BenchmarkSuite` validation for duplicate `(dataset, example_id)` pairs.
- Kept local `example_id` reuse legal across different datasets.
- Added a focused behavioral test for both duplicate rejection and cross-dataset reuse.

## Why

V1 benchmark rows and release traces identify examples by dataset plus example id. Allowing duplicate pairs inside one suite can make raw measurements ambiguous while aggregate report rows still look valid.

## Refactor Evidence

This is a narrow validation refactor around benchmark identity semantics. It preserves the existing public API and only rejects an invalid suite shape that previously produced ambiguous traces.

## Verification

```text
poetry run pytest tests/test_benchmarks.py::test_benchmark_suite_validates_identity_examples_and_datasets -q
1 passed in 0.06s

poetry run pytest tests/test_benchmarks.py tests/test_benchmark_runner.py -q
86 passed in 0.36s

poetry run pytest -q
1233 passed in 9.39s

poetry check
All set!

poetry run python -m compileall -q src tests
passed

git diff --check
passed
```
