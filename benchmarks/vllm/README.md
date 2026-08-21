# vLLM Benchmark Index

The current vLLM benchmark protocol is defined in the [benchmark root](../):
Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, shared GPU prefix
references, cold disk-to-GPU document-KV hydrate, `g6.8xlarge` (L4), 4 parallel
requests, N x 2k distinct documents per request, and forced 256-token decode.

Sanitized BF16 Vanilla-v2 three-arm canaries for baseline, an exact full-prefix
control, and independent pre-RoPE document segments live in
[`../appendix/representative-bf16-qwen3-4b-canaries/`](../appendix/representative-bf16-qwen3-4b-canaries/).
They use two HotpotQA examples with three repeats, isolated jobs, and request
parallelism 1. They are non-publication-qualified integration evidence, not
main-table measurements. Both 8k quality gates fail; the 16k gate passes. The
same folder contains a matched six-job cold-load ablation in which the direct
global-snapshot loader reduces unprofiled P50 TTFT by 60.22% at 8k and 61.70%
at 16k versus the forced legacy segment-remerge path. That ablation has no
prefetch events and makes no disk/prefill-overlap claim. The g6 and g5 nodes
also have different local-disk topologies, so their comparison is not a
GPU-only ablation.

Historical A10G warm-prefix canary evidence lives in
[`../appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](../appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/).
