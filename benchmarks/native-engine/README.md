# Native Engine Integration Evidence

This folder is the standalone, human-readable entry point for current Cachet
native vLLM and SGLang connector evidence. Dated subfolders are the report
pages to read and cite. These records support benchmark release readiness, but
they are not latency, throughput, or quality benchmark measurements.

## Current Evidence

| Backend | Databricks run | Source artifacts | Engine version | Payload mode | Copied tokens | Result |
| --- | --- | --- | --- | --- | ---: | --- |
| vLLM | `934698284395881` | [`2026-06-23-g6-l4-native-engine-probes/`](2026-06-23-g6-l4-native-engine-probes/) | `0.23.0` | merged | 48 | Provider-backed native probe succeeded |
| SGLang | `934698284395881` | [`2026-06-23-g6-l4-native-engine-probes/`](2026-06-23-g6-l4-native-engine-probes/) | `0.5.10.post1` | merged | 48 | Provider-backed dynamic HiCache probe succeeded |

## Scope

The native connector probes verify that Cachet uses engine-owned KV block
manager integration paths instead of a package-owned serving scheduler. They
also validate the connector action descriptors required by strict release
evidence.

For vLLM, latency and quality benchmark numbers live in `../vllm/`. For SGLang,
latency and quality benchmark numbers remain pending until a live SGLang
endpoint validates decode-time prefix binding with Cachet handoffs.

## Evidence Boundary

Use this folder for human review of runtime-integration evidence. Use the
linked Databricks folder for the sanitized
`document_kv.engine_kv_connector_probe.v1` and
`document_kv.engine_kv_connector_actions.v1` source records. Do not use
`../../pr-evidence/` as the benchmark report surface.
