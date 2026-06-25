# Native vLLM And SGLang Connector Evidence

This folder records provider-backed native connector probes for vLLM and
SGLang on the g6/L4 target. It proves integration with engine-owned KV block
manager paths; it is not latency, throughput, or quality evidence.

| Backend | Engine Version | Method | Copied Tokens | Outcome |
| --- | --- | --- | ---: | --- |
| vLLM | `0.23.0` | Vanilla external KV provider | 48 | Native probe succeeded |
| SGLang | `0.5.10.post1` | Dynamic HiCache provider | 48 | Native probe succeeded |

## Provenance

Sanitized source records are committed beside this README:

- `vllm_engine_probe.json`
- `vllm_connector_actions.json`
- `sglang_engine_probe.json`
- `sglang_connector_actions.json`
- `databricks_run_status.json`
