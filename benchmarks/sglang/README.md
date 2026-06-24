# SGLang Benchmark Status

This folder is the standalone, human-readable entry point for Cachet SGLang
benchmark status. There is not yet a published live SGLang latency or quality
benchmark result.

## Current Status

| Status | Databricks run | Source artifacts | Meaning |
| --- | --- | --- | --- |
| Latest failed live handoff smoke | `348824841142825` | [`2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/`](2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/) | The PR #467 wheel was present, import probe and request metadata bridge passed, and SGLang reported a positive 128-token external cache hit, but the later 46-key split query still missed and both live quality checks failed. It is not a benchmark result. |
| Failed live handoff smoke | `672750124167579` | [`2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/`](2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/) | The PR #466 wheel was present, but the request metadata bridge failed during import probe because the strict hash-tracking gate looked for `get_hash_str` in the wrong SGLang controller lifecycle hook. It is not a benchmark result. |
| Failed live handoff smoke | `73938470896039` | [`2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/`](2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/) | The PR #465 wheel was present, but the later 46-key storage query still logged `last_hash_present=False` because SGLang keeps the per-batch prior hash in a local `_storage_hit_query` variable. It is not a benchmark result. |
| Failed live handoff smoke | `476430354490832` | [`2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/`](2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/) | The PR #464 provider wheel was present and hydrated the first 128 runtime pages, but Cachet still missed the later 46-key storage query because SGLang chained it from runtime `last_hash`. It is not a benchmark result. |
| Failed live handoff smoke | `521023980659718` | [`2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/`](2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/) | SGLang launched with `wait_complete` prefetch and reported 128 cached tokens from Cachet handoff pages, but Cachet missed the later 46-key storage query and both live checks failed. It is not a benchmark result. |
| Failed live handoff smoke | `201402713679607` | [`2026-06-23-g6-l4-live-handoff-smoke/`](2026-06-23-g6-l4-live-handoff-smoke/) | Generated handoff preparation completed, but SGLang rejected the colon-containing served-model name before live requests. It is not a benchmark result. |
| Failed live handoff smoke | `13763847664432` | [`2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/`](2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/) | SGLang launched with the safe served-model name and request metadata bridge validation passed, but the cache arm used suffix-only runtime prompt text and did not answer from the generated prefix. It is not a benchmark result. |
| Failed live handoff smoke | `476596508869043` | [`2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/`](2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/) | SGLang launched with logical prompt text and answered both arms, but the cache arm reported zero cached tokens; the later cached-token line was ordinary SGLang prefix-cache reuse. It is not a benchmark result. |
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
generated-handoff Databricks smoke attempts are tracked in
[`2026-06-23-g6-l4-live-handoff-smoke/`](2026-06-23-g6-l4-live-handoff-smoke/)
and
[`2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/`](2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/),
plus the current zero-cache-hit blocker in
[`2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/`](2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/),
the partial page-binding blocker in
[`2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/`](2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/),
the chained hash-binding blocker in
[`2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/`](2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/),
the batch prior-hash metadata blocker in
[`2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/`](2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/),
the attach-time hash tracking gate in
[`2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/`](2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/),
and the current cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/`](2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/).
Use
`--baseline-only` for provider/server bring-up. For handoff-backed smoke,
prefer `--generate-live-handoff`, which generates the synthetic live Cachet
handoff and exact SGLang HiCache page keys inside the isolated SGLang runtime
before the server starts. The smoke now uses logical prompt text by default so
stock SGLang can compute the cached prefix page keys. The latest run also sets
`hicache_storage_prefetch_policy=wait_complete`, which allows SGLang to wait
for Cachet-backed page hydration before prefill. Recent failures show that
later split queries can be chained from SGLang's runtime hash state, and that
the relevant anchor is the first per-batch `prior_hash` passed to SGLang's
`get_hash_str`, not merely `operation.last_hash`. The latest run shows that
attach-time hash tracking installs successfully and that SGLang can report a
positive cache-arm external hit, but the later 46-key query still missed and
both live quality checks failed. A publishable live run must hydrate all
matching generated page-key chunks and record a positive SGLang cached-token
validation before it can be treated as successful. That validation must identify
the cache-arm request itself by its prompt-token total, rather than warmup
requests or a later baseline request that benefits from ordinary SGLang
prefix-cache reuse. Manual handoff inputs remain supported when callers already
have a validated SGLang handoff plus
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
- Split SGLang HiCache storage queries hydrate every matching generated handoff
  page chunk, including later chunks anchored by the batch `prior_hash` used to
  compute SGLang's chained runtime keys.
- The cache-arm request reports positive SGLang cached-token validation.
- The cached-token validation is matched to the cache-arm request, not to server
  warmup traffic or later baseline prefix-cache reuse.
- Latency, throughput, and quality measurements are recorded with the same
  benchmark schema used by the vLLM report.

## Evidence Boundary

The native probe source records under `../databricks/` prove that Cachet can
wire provider-backed dynamic HiCache integration in the target runtime. They
must not be cited as SGLang latency, throughput, or quality benchmark results.
Do not use `../../pr-evidence/` as the benchmark report surface.
