# Seed Benchmark Shuffle By Dataset Example

## Summary

- Updated seeded benchmark arm shuffling to derive its deterministic seed from `(dataset, example_id)`.
- Added a regression test for suites that reuse the same local `example_id` across different V1 datasets.

## Why

`BenchmarkSuite` now allows local example ids to repeat across datasets. The benchmark runner still seeded shuffled arm order with only `example_id`, which made cross-dataset examples with the same local id use identical arm ordering. Including the dataset keeps deterministic shuffling while preserving per-dataset traceability.

## Refactor Evidence

This is a narrow behavior-preserving hardening change for the generalized V1 benchmark identity contract. It does not change public APIs or benchmark result schemas.

## Verification

```text
poetry run pytest tests/test_benchmark_runner.py::test_seeded_shuffle_uses_dataset_and_example_identity tests/test_benchmark_runner.py::test_run_benchmark_suite_supports_repeats_and_seeded_shuffle -q
2 passed in 0.07s

poetry run pytest tests/test_benchmark_runner.py tests/test_benchmarks.py -q
87 passed in 0.38s

poetry run pytest -q
1234 passed in 9.09s

poetry check
All set!

poetry run python -m compileall -q src tests
passed

git diff --check
passed
```
