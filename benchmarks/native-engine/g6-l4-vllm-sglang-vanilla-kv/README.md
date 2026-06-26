# Native vLLM And SGLang Connector Evidence

Provider-backed native connector probes for vLLM and SGLang on g6/L4. This is
engine-integration and copied-byte evidence, not latency, throughput, or
quality evidence.

## Experimental Setup

| Backend | Engine version | Model fixture | Hardware | Method | Fixture scope | Evidence file |
| --- | --- | --- | --- | --- | --- | --- |
| vLLM | `0.23.0` | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV provider | 48 copied tokens | [`vllm_engine_probe.json`](vllm_engine_probe.json) |
| SGLang | `0.5.10.post1` | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Dynamic HiCache provider | 48 copied tokens | [`sglang_engine_probe.json`](sglang_engine_probe.json) |

## Copied KV Footprint

| Backend | Copied tokens | Copied bytes | Bytes per token | Copied segments | Total blocks | Estimated GPU bytes | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM | 48 | 3,538,944 | 73,728 | 3 | 3 | not measured | Native probe succeeded |
| SGLang | 48 | 3,538,944 | 73,728 | 3 | 3 | not measured | Native probe succeeded |

## Not Measured

| Metric | Status |
| --- | --- |
| p50 TTFT / time-to-completion | not measured |
| Answer quality | not measured |
| Peak GPU memory | not measured |
| CPU RSS | not measured |
| Serving cache footprint | not measured |

## Provenance

Sanitized source records are committed beside this README:

- [`vllm_engine_probe.json`](vllm_engine_probe.json)
- [`vllm_connector_actions.json`](vllm_connector_actions.json)
- [`sglang_engine_probe.json`](sglang_engine_probe.json)
- [`sglang_connector_actions.json`](sglang_connector_actions.json)
- [`databricks_run_status.json`](databricks_run_status.json)
