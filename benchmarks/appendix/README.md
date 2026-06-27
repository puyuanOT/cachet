# Benchmark Appendix

This appendix preserves benchmark evidence for audit and historical comparison.
Use the [benchmark root](../) for the canonical primary table and ablation
tables.

| Folder | Purpose |
| --- | --- |
| [`primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/`](primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/) | Current sanitized main-table evidence with forced-256 latency, natural-stop quality, and Cachet provider telemetry |
| [`primary-table-v2-vllm-qwen3-4b-g5-a10g-disk-cache/`](primary-table-v2-vllm-qwen3-4b-g5-a10g-disk-cache/) | Superseded primary-table evidence without forced-256 validation on every row |
| [`primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/`](primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/) | Superseded baseline and full-logical-prompt vLLM Cachet handoff-path evidence from the earlier prepared suite |
| [`runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/`](runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/) | Failed runtime-prompt vLLM canary documenting why the current provider uses logical prompt text for external-KV loads |
| [`existing-results/`](existing-results/) | Committed evidence produced under non-primary benchmark configurations |

Use the [benchmark root](../) for the main performance table and ablation
tables.
