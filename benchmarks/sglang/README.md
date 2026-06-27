# SGLang Benchmark Index

The primary benchmark table is fixed to vLLM on `g5.8xlarge`; see the
[benchmark root](../). SGLang appears in the serving-platform ablation table
with blank metric cells until a matching run exists.

Existing SGLang evidence:

| Appendix result | Evidence summary |
| --- | --- |
| [`../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-prepared/`](../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-prepared/) | Prepared V1 correctness and cache-hit evidence; no latency improvement observed |
| [`../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/`](../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/) | Minimal synthetic cache-hit validation; no latency improvement observed |

Failed and superseded SGLang smoke attempts are preserved under
[`../../docs/release-ops/benchmark-archive/sglang-smoke/`](../../docs/release-ops/benchmark-archive/sglang-smoke/).
They are useful for maintainers, but they are not public benchmark results.
