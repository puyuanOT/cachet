# Benchmark Appendix

This appendix preserves benchmark evidence for audit and historical comparison.
Use the [benchmark root](../) for the canonical primary table and ablation
tables.

| Folder | Purpose |
| --- | --- |
| [`primary-table-v2-vllm-qwen3-4b-g5-a10g-disk-cache/`](primary-table-v2-vllm-qwen3-4b-g5-a10g-disk-cache/) | Current sanitized main-table evidence for baseline and Cachet + vanilla KV at 8k, 16k, and 32k |
| [`primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/`](primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/) | Superseded baseline and full-logical-prompt vLLM Cachet handoff-path evidence from the earlier prepared suite |
| [`runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/`](runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/) | Failed runtime-prompt vLLM canary documenting why the current provider uses logical prompt text for external-KV loads |
| [`existing-results/`](existing-results/) | Committed evidence produced under non-primary benchmark configurations |

Use the [benchmark root](../) for the main performance table and ablation
tables.
