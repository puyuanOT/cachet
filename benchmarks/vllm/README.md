# vLLM Benchmark Index

The primary vLLM comparison is defined in the [benchmark root](../). The
fixed target configuration is Qwen3-4B-Instruct on vLLM, `g5.8xlarge`, 8
parallel requests, 256 emitted tokens, and disk-resident Cachet KV.

Existing vLLM evidence that does not match the primary-table configuration:

| Appendix result | Status |
| --- | --- |
| [`../appendix/existing-results/vllm-qwen3-4b-g6-l4-vanilla-kv/`](../appendix/existing-results/vllm-qwen3-4b-g6-l4-vanilla-kv/) | Existing g6/L4 vanilla KV latency evidence |
| [`../appendix/existing-results/vllm-qwen3-4b-g5-a10g-vanilla-kv/`](../appendix/existing-results/vllm-qwen3-4b-g5-a10g-vanilla-kv/) | Existing g5/A10G vanilla KV compatibility evidence |

Use [`../databricks/`](../databricks/) for sanitized audit mirrors.
