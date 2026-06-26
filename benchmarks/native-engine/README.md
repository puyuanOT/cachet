# Native Engine Integration Evidence

These probes show that Cachet can hand external KV to established serving
engines through native KV-transfer/cache surfaces. They are integration and
footprint evidence, not latency or quality benchmarks.

## Experimental Setup

| Result | Backend | Engine version | Model fixture | Hardware | Method | Fixture scope | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`g6-l4-vllm-sglang-vanilla-kv/`](g6-l4-vllm-sglang-vanilla-kv/) | vLLM | `0.23.0` | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV provider | 48 copied tokens | [`g6-l4-vllm-sglang-vanilla-kv/vllm_engine_probe.json`](g6-l4-vllm-sglang-vanilla-kv/vllm_engine_probe.json) |
| [`g6-l4-vllm-sglang-vanilla-kv/`](g6-l4-vllm-sglang-vanilla-kv/) | SGLang | `0.5.10.post1` | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Dynamic HiCache provider | 48 copied tokens | [`g6-l4-vllm-sglang-vanilla-kv/sglang_engine_probe.json`](g6-l4-vllm-sglang-vanilla-kv/sglang_engine_probe.json) |

## Copied KV Footprint

| Backend | Copied tokens | Copied bytes | Bytes per token | Copied segments | Total blocks | Estimated GPU bytes | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM | 48 | 3,538,944 | 73,728 | 3 | 3 | not measured | Provider-backed native probe succeeded |
| SGLang | 48 | 3,538,944 | 73,728 | 3 | 3 | not measured | HiCache-backed native probe succeeded |

## What Is Not Measured

| Metric | Status |
| --- | --- |
| p50 TTFT / time-to-completion | not measured |
| Answer quality | not measured |
| Peak GPU memory | not measured |
| CPU RSS | not measured |
| Serving cache footprint | not measured |

For performance numbers, use [`../vllm/`](../vllm/) and [`../sglang/`](../sglang/).
