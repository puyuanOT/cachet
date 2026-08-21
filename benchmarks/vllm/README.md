# vLLM Benchmark Index

The current vLLM benchmark protocol is defined in the [benchmark root](../):
Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, shared GPU prefix
references, cold disk-to-GPU document-KV hydrate, `g6.8xlarge` (L4), 4 parallel
requests, N x 2k distinct documents per request, and forced 256-token decode.

The current Q4/Q8 Baseline-versus-Vanilla and ablation measurements are backed
by [compact sanitized evidence](../appendix/main-vanilla-descriptive-evidence/).
They are descriptive/nonpublication-qualified: the isolated Baseline raw
record structurally fails the generic cache-arm/resource-schema canary gate.

Sanitized BF16 Vanilla three-arm canaries for baseline, an exact full-prefix
control, and independent pre-RoPE document segments live in
[`../appendix/representative-bf16-qwen3-4b-canaries/`](../appendix/representative-bf16-qwen3-4b-canaries/).
They use two HotpotQA examples with three repeats, isolated jobs, and request
parallelism 1. They are non-publication-qualified integration evidence, not
main-table measurements. Both 8k quality gates fail; the 16k gate passes. The
same folder contains a matched eight-job cold-load ablation in which the direct
global-snapshot loader reduces unprofiled P50 TTFT by 60.22% at 8k, 61.70% at
16k, and 61.81% at 32k versus the forced legacy segment-remerge path. The 32k
pair is internally matched at `gpu_memory_utilization=0.70`, while the 8k and
16k pairs use `0.85`, so the cross-size trend is not an identical-engine-memory
scaling experiment. That ablation has no prefetch events and makes no
disk/prefill-overlap claim. The g6 and g5 nodes also have different local-disk
topologies, so their comparison is not a GPU-only ablation.

Historical A10G warm-prefix canary evidence lives in
[`../appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](../appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/).
