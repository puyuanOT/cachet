# SGLang Benchmark Index

The current main latency/resource table is fixed to vLLM. SGLang appears only
in the serving-platform ablation in the [benchmark root](../), with blank cells
until a matching Q4-weight + Q8-document-KV run exists.

A sanitized g6/L4 BF16 native-handoff smoke is committed under
[`../appendix/representative-bf16-qwen3-4b-canaries/`](../appendix/representative-bf16-qwen3-4b-canaries/).
Its configured profile reserves 4k input and 32 output tokens, but the two
measured requests per arm actually used 205 prompt tokens and completed with 7
tokens. It validates native cache-path execution only; it is not a 4k-to-32
latency result, a resource claim, or a main-table serving-platform ablation.

Historical SGLang smoke attempts remain outside `benchmarks/` under
[`../../docs/release-ops/benchmark-archive/`](../../docs/release-ops/benchmark-archive/)
for maintainer debugging.
