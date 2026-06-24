# SGLang Benchmark Status

This folder is the standalone, human-readable entry point for Cachet SGLang
benchmark status. There is not yet a published live SGLang latency or quality
benchmark result.

## Current Status

| Status | Databricks run | Source artifacts | Meaning |
| --- | --- | --- | --- |
| Failed live handoff smoke | `201402713679607` | [`2026-06-23-g6-l4-live-handoff-smoke/`](2026-06-23-g6-l4-live-handoff-smoke/) | Generated handoff preparation completed, but SGLang rejected the colon-containing served-model name before live requests. It is not a benchmark result. |
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
the runtime `prefix_keys` and batch hashes. `sglang_kv_injection.hicache_keys`
and `cachet-benchmark-handoff-manifest --sglang-hicache-page-keys-json-template`
provide the local page-key metadata path for future live runs. Cachet also
installs `sglang_kv_injection.sglang_request_metadata_bridge` from the dynamic
HiCache backend so pinned SGLang request `custom_params` reach
`HiCacheStorageExtraInfo.extra_info` during storage hit queries and page
transfers. The runtime preflight records that bridge separately from upstream
SGLang source detection and can report `live_request_metadata_bridge_ok=true`
when all patch points install.

Those helpers are a live readiness path, not a benchmark result. The current
generated-handoff Databricks smoke is tracked in
[`2026-06-23-g6-l4-live-handoff-smoke/`](2026-06-23-g6-l4-live-handoff-smoke/).
Use
`--baseline-only` for provider/server bring-up. For handoff-backed smoke,
prefer `--generate-live-handoff`, which generates the synthetic live Cachet
handoff and exact SGLang HiCache page keys inside the isolated SGLang runtime
before the server starts. Manual handoff inputs remain supported when callers
already have a validated SGLang handoff plus
`document_kv.sglang_hicache_page_keys` metadata. This report remains pending
until a Databricks g6/L4 or g5/A10G run writes `sglang-live-smoke.json` with a
passing handoff-backed cache arm.

## Benchmark Gate

Treat SGLang benchmark publication as pending until all of these are true:

- A real SGLang serving endpoint is launched on the target Databricks hardware.
- The endpoint receives Cachet `kv_transfer_params` from a generated live
  handoff or an equivalent validated handoff.
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
