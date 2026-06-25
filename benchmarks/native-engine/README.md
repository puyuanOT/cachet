# Native Engine Integration Evidence

This folder proves Cachet can hand external KV to established serving engines
through their native KV-transfer surfaces. It is integration evidence, not a
latency or throughput benchmark.

## Current Evidence

| Result | Backend | Engine Version | Hardware | Method | Copied Tokens | Outcome |
| --- | --- | --- | --- | --- | ---: | --- |
| [`g6-l4-vllm-sglang-vanilla-kv/`](g6-l4-vllm-sglang-vanilla-kv/) | vLLM | `0.23.0` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV provider | 48 | Provider-backed native probe succeeded |
| [`g6-l4-vllm-sglang-vanilla-kv/`](g6-l4-vllm-sglang-vanilla-kv/) | SGLang | `0.5.10.post1` | AWS g6/L4, `g6.8xlarge` | Dynamic HiCache provider | 48 | Provider-backed native probe succeeded |

## Interpretation

These probes verify that Cachet integrates with engine-owned KV block managers
instead of shipping a custom serving scheduler. For performance numbers, use
[`../vllm/`](../vllm/) and [`../sglang/`](../sglang/).
