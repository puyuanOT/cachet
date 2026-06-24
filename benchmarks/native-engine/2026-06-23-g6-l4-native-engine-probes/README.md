# AWS g6/L4 Native Engine Probes

This standalone report records provider-backed native connector evidence for
vLLM and SGLang on the strict Databricks g6/L4 target. It supports benchmark
readiness, but it is not a latency, throughput, or quality benchmark result.

| Field | Value |
| --- | --- |
| Date | 2026-06-23 |
| Scope | Native engine integration evidence |
| Target | `aws-g6-l4` / `g6.8xlarge` |
| Databricks run | `934698284395881` |
| Task count | 2 |
| Result | Integration evidence; both tasks terminated `SUCCESS` |

## Human Result

Both native probes succeeded with provider-backed Cachet wiring against
engine-owned KV block manager paths. Each probe copied 48 tokens and 3,538,944
bytes through the native connector test path.

| Backend | Engine version | Payload mode | Copied tokens | Copied bytes | Provider factory |
| --- | --- | --- | ---: | ---: | --- |
| vLLM | `0.23.0` | merged | 48 | 3538944 | `vllm_kv_injection.vllm_native_provider:build_document_kv_provider` |
| SGLang | `0.5.10.post1` | merged | 48 | 3538944 | `sglang_kv_injection.sglang_dynamic_backend:build_document_kv_hicache_provider` |

## Scope

These records prove native runtime integration paths, not serving benchmark
latency or quality. vLLM latency and quality results live under
`../../vllm/`; the current prepared live SGLang V1 benchmark result lives under
`../../sglang/2026-06-24-g6-l4-prepared-v1-release-suite-success/`.

## Source Artifacts

The sanitized source records live in
[`../../databricks/2026-06-23-g6-l4-native-engine-probes/`](../../databricks/2026-06-23-g6-l4-native-engine-probes/):

- `databricks_run_status.json`
- `vllm_engine_probe.json`
- `sglang_engine_probe.json`
- `vllm_connector_actions.json`
- `sglang_connector_actions.json`

## Artifact Boundary

This folder is the human-readable integration-evidence report. Keep raw
Databricks Jobs API responses, tokens, wheels, logs, generated datasets, and
local scratch output out of this tree.
