# SGLang Benchmark Index

The current main latency/resource table is fixed to vLLM. SGLang appears only
in the serving-platform ablation in the [benchmark root](../), where it is
explicitly `N/A` because the Q8 pre-RoPE serving path is not implemented.

A sanitized g6/L4 BF16 native-handoff smoke is committed under
[`../appendix/representative-bf16-qwen3-4b-canaries/`](../appendix/representative-bf16-qwen3-4b-canaries/).
Its configured profile reserves 4k input and 32 output tokens, but the two
measured requests per arm actually used 205 prompt tokens and completed with 7
tokens. Both cache requests validate 176 cached tokens under the Vanilla
pre-RoPE position contract, but the cache arm is slower in this smoke (P50 TTFT
4.570751s versus 2.911591s baseline). It validates native cache-path execution
only; it is not a 4k-to-32 latency result, a resource claim, or a main-table
serving-platform ablation.

Historical SGLang smoke attempts remain outside `benchmarks/` under
[`../../docs/release-ops/benchmark-archive/`](../../docs/release-ops/benchmark-archive/)
for maintainer debugging.
