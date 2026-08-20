# Benchmark Appendix

This appendix holds sanitized engineering canaries and historical benchmark evidence
that do not populate the current publication tables. The current
latency/resource, dataset-score, and ablation tables live in the
[benchmark root](../). The existing representative BF16 records were produced
by superseded Vanilla-v1 post-RoPE code and are retained only until the v2
refresh lands. The
historical evidence predates the current g6/L4, request-parallelism-4,
N x 2k-document cold-hydrate protocol (it is A10G warm-prefix canary data) and
is retained for provenance only.

| Folder | Purpose |
| --- | --- |
| [`representative-bf16-qwen3-4b-canaries/`](representative-bf16-qwen3-4b-canaries/) | Historical Vanilla-v1 post-RoPE BF16 canaries awaiting replacement by fresh Vanilla-v2 evidence |
| [`current-q4-q8-vllm-qwen3-4b-g5-a10g/`](current-q4-q8-vllm-qwen3-4b-g5-a10g/) | Historical A10G warm-prefix Qwen3-4B Q4-weight, Q8-document-KV vLLM canary evidence; predates the current cold-hydrate protocol |
