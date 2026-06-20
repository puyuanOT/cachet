# Harden Benchmark Generation Schema

This PR tightens the benchmark engine-output boundary.

`BenchmarkGeneration` now validates that generated benchmark output is safe to serialize and summarize:

- `output_text` must be a string, while empty strings remain valid.
- Prompt and completion token counts must be non-negative integers.
- TTFT and time-to-completion must be finite, non-negative numbers, with completion time greater than or equal to TTFT.
- `metadata` must be a string-to-string mapping.

Because engines construct `BenchmarkGeneration` inside `generate()`, invalid generation objects are caught by the benchmark runner and recorded as error measurements rather than aborting the whole benchmark run.
