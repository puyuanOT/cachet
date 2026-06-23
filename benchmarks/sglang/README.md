# SGLang Benchmark Status

This folder is the standalone, human-readable entry point for Cachet SGLang
benchmark status. There is not yet a published live SGLang latency or quality
benchmark result.

## Current Status

| Status | Databricks run | Source artifacts | Meaning |
| --- | --- | --- | --- |
| Pending latency and quality benchmark | Not available yet | Not available yet | A live SGLang endpoint still needs to validate decode-time prefix binding with Cachet handoffs before Cachet can publish SGLang latency, throughput, or quality numbers. |
| Native HiCache integration evidence | `934698284395881` | [`../databricks/2026-06-23-g6-l4-native-engine-probes/`](../databricks/2026-06-23-g6-l4-native-engine-probes/) | Provider-backed native probe and connector-action records succeeded on AWS g6/L4, but these records are integration evidence, not benchmark measurements. |

## Live Smoke Helper

`document_kv_cache.sglang_smoke` and
`document_kv_cache.databricks_sglang_smoke_job` provide the current live SGLang
readiness path. The smoke launches pinned SGLang with Cachet's dynamic HiCache
provider config, validates the provider factory, and runs a full-prompt
baseline. Cachet's built-in provider can hydrate validated handoff payload
pages under SGLang runtime HiCache keys when request context reaches the
provider with explicit `document_kv.sglang_hicache_page_keys` metadata matching
the runtime `prefix_keys` and batch hashes. Handoff-backed cache-arm execution
is intentionally blocked before server launch because the pinned SGLang OpenAI
path carries `custom_params` on sampling params without passing Cachet
`kv_transfer_params` into `HiCacheStorageExtraInfo.extra_info`. The runtime
preflight exposes this as `live_request_metadata_bridge_ok=false`.

Those helpers are a provider/server bring-up path, not a benchmark result. Use
`--baseline-only` until the live request metadata bridge is available. This
report remains pending until a Databricks g6/L4 or g5/A10G run writes
`sglang-live-smoke.json` with a passing handoff-backed cache arm.

## Benchmark Gate

Treat SGLang benchmark publication as pending until all of these are true:

- A real SGLang serving endpoint is launched on the target Databricks hardware.
- The endpoint receives Cachet `kv_transfer_params` from a generated handoff.
- Cachet handoff params include SGLang page-key metadata that matches the
  runtime HiCache hash chain.
- The SGLang runtime preflight reports `live_request_metadata_bridge_ok=true`.
- Decode-time prefix binding is validated against the live endpoint.
- Latency, throughput, and quality measurements are recorded with the same
  benchmark schema used by the vLLM report.

## Evidence Boundary

The native probe source records under `../databricks/` prove that Cachet can
wire provider-backed dynamic HiCache integration in the target runtime. They
must not be cited as SGLang latency, throughput, or quality benchmark results.
Do not use `../../pr-evidence/` as the benchmark report surface.
