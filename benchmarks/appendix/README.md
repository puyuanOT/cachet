# Benchmark Appendix

This appendix holds sanitized engineering canaries and historical benchmark
evidence that do not populate the current publication tables. The current
latency/resource, dataset-score, and ablation tables live in the
[benchmark root](../). The representative BF16 folder contains current
Vanilla-v2 pre-RoPE canaries and cold-load optimization evidence. The separate
A10G warm-prefix folder is historical evidence that predates the current
g6/L4, request-parallelism-4, N x 2k-document cold-hydrate protocol and is
retained for provenance only.

| Folder | Purpose |
| --- | --- |
| [`representative-bf16-qwen3-4b-canaries/`](representative-bf16-qwen3-4b-canaries/) | Vanilla-v2 BF16 vLLM canaries, SGLang native-handoff smoke, and an eight-job direct-versus-legacy cold-load ablation through 32k; non-publication-qualified |
| [`current-q4-q8-vllm-qwen3-4b-g5-a10g/`](current-q4-q8-vllm-qwen3-4b-g5-a10g/) | Historical A10G warm-prefix Qwen3-4B Q4-weight, Q8-document-KV vLLM canary evidence; predates the current cold-hydrate protocol |
