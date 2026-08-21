# Benchmark Appendix

This appendix holds sanitized descriptive engineering evidence, canaries, and
historical benchmark evidence. The current latency/resource, dataset-score,
and ablation tables live in the [benchmark root](../). The Q4/Q8 descriptive
folder contains the compact evidence behind the populated current tables; its
isolated Baseline raw record does not pass the generic canary gate, so those
results remain nonpublication-qualified. The representative BF16 folder
contains separate Vanilla pre-RoPE canaries and cold-load optimization
evidence. The A10G warm-prefix folder is historical evidence that predates the
current g6/L4, request-parallelism-4, N x 2k-document cold-hydrate protocol and
is retained for provenance only.

| Folder | Purpose |
| --- | --- |
| [`main-vanilla-descriptive-evidence/`](main-vanilla-descriptive-evidence/) | Current compact Q4/Q8 table evidence, including main latency, precision, storage, hardware, serving-platform, and five-example score diagnostics; descriptive/nonpublication-qualified |
| [`representative-bf16-qwen3-4b-canaries/`](representative-bf16-qwen3-4b-canaries/) | Vanilla BF16 vLLM canaries, SGLang native-handoff smoke, and an eight-job direct-versus-legacy cold-load ablation through 32k; non-publication-qualified |
| [`current-q4-q8-vllm-qwen3-4b-g5-a10g/`](current-q4-q8-vllm-qwen3-4b-g5-a10g/) | Historical A10G warm-prefix Qwen3-4B Q4-weight, Q8-document-KV vLLM canary evidence; predates the current cold-hydrate protocol |
