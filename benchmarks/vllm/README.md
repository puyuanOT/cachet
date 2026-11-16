# vLLM Benchmark Index

The published campaign ran on vLLM 0.30.0, ported from the frozen vLLM 0.27.1
publication design described in the [benchmark root](../). Baseline and
Vanilla KV cover 8k, 16k, and 32k with closed-loop concurrency 1, 2, and 4
across five matched deployment blocks.

No pre-reset numeric result is carried forward. The sealed campaign report and
its passing publication-gate pair are in
[`../appendix/vllm-0271-publication-v1/`](../appendix/vllm-0271-publication-v1/).
Answer quality is reported by
[Quality Benchmark v2](../README.md#quality-benchmark-v2), and the auxiliary
latency contrasts by the
[Auxiliary Latency Re-run](../README.md#auxiliary-latency-re-run). The
benchmark root is the canonical current-state surface.
