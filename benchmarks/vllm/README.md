# vLLM Benchmark Index

The main vLLM comparison now lives in the paper-style benchmark appendix at
[benchmark root](../). The fixed target configuration is Qwen3-4B-Instruct
on vLLM, `g5.8xlarge`, 8 parallel requests, 256 emitted tokens, and disk-backed
Cachet methods.

Existing vLLM evidence that does not match that fixed configuration moved to:

| Appendix result | Status |
| --- | --- |
| [`../appendix/existing-results/vllm-qwen3-4b-g6-l4-vanilla-kv/`](../appendix/existing-results/vllm-qwen3-4b-g6-l4-vanilla-kv/) | Prior g6/L4 vanilla KV speedup evidence |
| [`../appendix/existing-results/vllm-qwen3-4b-g5-a10g-vanilla-kv/`](../appendix/existing-results/vllm-qwen3-4b-g5-a10g-vanilla-kv/) | Prior g5/A10G vanilla KV compatibility evidence |

Use [`../databricks/`](../databricks/) for sanitized audit mirrors.
