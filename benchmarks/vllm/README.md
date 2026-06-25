# vLLM Benchmarks

These are the current Cachet performance results for vLLM. They compare the
standard no-cache prefill baseline with Cachet's vanilla external KV cache arm.

## Results

| Result | Model | Hardware | Method | Measurements | TTFT Speedup | Time-To-Completion Speedup | Quality Delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [`qwen3-4b-g6-l4-vanilla-kv/`](qwen3-4b-g6-l4-vanilla-kv/) | Qwen3 4B Instruct | AWS g6/L4, `g6.8xlarge` | Vanilla external KV | 24 | 5.27x-6.97x | 1.74x-2.25x | 0.0 |
| [`qwen3-4b-g5-a10g-vanilla-kv/`](qwen3-4b-g5-a10g-vanilla-kv/) | Qwen3 4B Instruct | AWS g5/A10G, `g5.8xlarge` | Vanilla external KV | 24 | 4.66x-6.04x | 2.04x-2.67x | 0.0 |

The g6/L4 result is the primary target. The g5/A10G result is compatibility
evidence for users running g5 clusters.

## What This Proves

- Cachet can beat vLLM no-cache prefill on the current four-dataset V1 suite.
- The published method is vanilla external KV cache handoff.
- The current public model coverage is Qwen3 4B Instruct.
- Answer quality stayed unchanged in the reported comparisons.

## What This Does Not Prove

- It does not benchmark KV Packet yet.
- It does not claim performance for other models or hardware until matching
  folders exist.
- It does not describe SGLang performance; use [`../sglang/`](../sglang/) for
  that status.

Databricks run IDs and sanitized run-status mirrors live under
[`../databricks/`](../databricks/) for audits.
