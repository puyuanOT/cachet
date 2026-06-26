# SGLang Benchmark Index

The main benchmark table is now fixed to vLLM on `g5.8xlarge`; see
[`../current/`](../current/). SGLang appears in the serving-platform ablation
table with blank cells until a matching run exists.

Existing SGLang evidence moved to:

| Appendix result | Status |
| --- | --- |
| [`../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-prepared/`](../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-prepared/) | Prepared V1 correctness/cache-hit evidence; no speedup |
| [`../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/`](../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/) | Tiny synthetic cache-hit check; no speedup |

Failed and superseded SGLang smoke attempts are preserved under
[`../../docs/release-ops/benchmark-archive/sglang-smoke/`](../../docs/release-ops/benchmark-archive/sglang-smoke/).
They are useful for maintainers, but they are not public benchmark results.
