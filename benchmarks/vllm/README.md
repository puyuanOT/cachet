# vLLM Benchmark Index

The current vLLM benchmark protocol is defined in the [benchmark root](../):
Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, shared GPU prefix
references, cold disk-to-GPU document-KV hydrate, `g5.8xlarge`, 8 parallel
requests, and forced 256-token decode.

Warm-prefix canary evidence lives in
[`../appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](../appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/).
