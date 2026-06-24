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
the current prepared live V1 benchmark lives in
`../sglang/2026-06-24-g6-l4-prepared-v1-release-suite-success/`; native-engine
records remain integration evidence only.

## Evidence Boundary

Use this folder for human review of runtime-integration evidence. The dated
report folder includes the sanitized
`document_kv.engine_kv_connector_probe.v1` and
`document_kv.engine_kv_connector_actions.v1` records it cites. Use
`../databricks/` as a release-source mirror, not as the only benchmark surface.
Do not use `../../docs/release-ops/pr-evidence/` as the benchmark report
surface.
