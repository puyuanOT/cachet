# SGLang Benchmark Status

This folder is the standalone, human-readable entry point for Cachet SGLang
benchmark status. There is now a successful live SGLang handoff smoke on
g6/L4, but there is not yet a published SGLang latency or throughput benchmark
suite.

## Current Status

| Status | Databricks run | Source artifacts | Meaning |
| --- | --- | --- | --- |
| Latest successful live handoff smoke | `134006212072875` | [`2026-06-24-g6-l4-live-handoff-smoke-baseline-isolated-success/`](2026-06-24-g6-l4-live-handoff-smoke-baseline-isolated-success/) | The PR #490 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation used deterministic no-thinking controls, the smoke ran a clean baseline first, `/flush_cache` returned HTTP 200 before the cache arm, SGLang reported a full 175-token cache-arm hit covering the generated prefix, baseline and cache-arm outputs both returned `otkv7391`, `/flush_cache` returned HTTP 200 before the canary, and the canary produced `cachet-green` with zero cached tokens. This is a readiness pass, not a latency or throughput benchmark suite. |
| Previous failed live handoff smoke | `655273897262076` | [`2026-06-24-g6-l4-live-handoff-smoke-canary-flush-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-canary-flush-cache-hit-quality-failure/) | The PR #488 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation used the tokenizer chat template with deterministic no-thinking controls, SGLang reported a full 175-token cache-arm hit covering that generated prefix, `/flush_cache` returned HTTP 200 before the post-live-check canary, and the canary passed with zero cached tokens, but baseline and cache-arm quality still failed. It is not a benchmark result. |
| Failed live handoff smoke | `419314952937106` | [`2026-06-24-g6-l4-live-handoff-smoke-canary-after-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-canary-after-cache-hit-quality-failure/) | The PR #486 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation used the tokenizer chat template with deterministic no-thinking controls, SGLang accepted `--attention-backend triton`, `--sampling-backend pytorch`, and `--enable-deterministic-inference`, the model-quality canary now ran after cache/baseline checks, and SGLang reported a full 175-token cache-arm hit covering that generated prefix, but baseline, cache-arm, and post-live-check canary quality all failed. It is not a benchmark result. |
| Failed live handoff smoke | `672927118707206` | [`2026-06-24-g6-l4-live-handoff-smoke-minimal-no-thinking-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-minimal-no-thinking-cache-hit-quality-failure/) | The PR #482 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation used the tokenizer chat template with deterministic no-thinking controls, SGLang accepted `--attention-backend triton`, `--sampling-backend pytorch`, and `--enable-deterministic-inference`, the live request used a minimal no-thinking body without sampling extras, and SGLang reported a positive 175-token cache-arm hit covering that generated prefix, but both live quality checks returned the same repeated filler text. It is not a benchmark result. |
| Failed live handoff smoke | `585529688094161` | [`2026-06-24-g6-l4-live-handoff-smoke-triton-deterministic-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-triton-deterministic-cache-hit-quality-failure/) | The PR #480 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation used the tokenizer chat template with deterministic no-thinking controls, SGLang accepted `--attention-backend triton`, `--sampling-backend pytorch`, and `--enable-deterministic-inference`, and SGLang reported a positive 175-token cache-arm hit covering that generated prefix, but both live quality checks returned the same repeated filler text. It is not a benchmark result. |
| Failed live handoff smoke | `647563677081667` | [`2026-06-24-g6-l4-live-handoff-smoke-deterministic-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-deterministic-cache-hit-quality-failure/) | The PR #478 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation used the tokenizer chat template with deterministic no-thinking controls, the live request used `/v1/chat/completions` with `temperature=0.0`, and SGLang reported a positive 175-token cache-arm hit covering that generated prefix, but both live quality checks returned the same repeated filler text. It is not a benchmark result. |
| Failed live handoff smoke | `417035094778538` | [`2026-06-24-g6-l4-live-handoff-smoke-no-thinking-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-no-thinking-cache-hit-quality-failure/) | The PR #476 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation used the tokenizer chat template with no-thinking controls, the live request used `/v1/chat/completions`, and SGLang reported a positive 175-token cache-arm hit covering that generated prefix, but both live quality checks returned repeated filler text. It is not a benchmark result. |
| Failed live handoff smoke | `163920824964705` | [`2026-06-24-g6-l4-live-handoff-smoke-chat-completions-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-chat-completions-cache-hit-quality-failure/) | The PR #473 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation used the tokenizer chat template, the live request used `/v1/chat/completions`, and SGLang reported a positive 175-token cache-arm hit covering that generated prefix, but both live quality checks returned repeated filler text. It is not a benchmark result. |
| Failed live handoff smoke | `585131634686228` | [`2026-06-24-g6-l4-live-handoff-smoke-qwen-sampling-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-qwen-sampling-cache-hit-quality-failure/) | The PR #471 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation produced a token-stable 175-token prefix, Qwen sampling parameters reached the live request path, and SGLang reported a positive cache-arm hit covering that generated prefix, but both live quality checks returned repeated filler text. It is not a benchmark result. |
| Failed live handoff smoke | `897276220223990` | [`2026-06-24-g6-l4-live-handoff-smoke-qwen-chat-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-qwen-chat-cache-hit-quality-failure/) | The PR #470 wheel was present, import probe and request metadata bridge passed, generated Qwen-chat handoff creation produced a token-stable 175-token prefix, and SGLang reported a positive cache-arm hit covering that generated prefix, but both live quality checks returned repeated filler text. It is not a benchmark result. |
| Failed live handoff smoke | `995284076545208` | [`2026-06-24-g6-l4-live-handoff-smoke-token-stable-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-token-stable-cache-hit-quality-failure/) | The PR #469 wheel was present, import probe and request metadata bridge passed, generated handoff creation produced a token-stable 149-token prefix, and SGLang reported a positive cache-arm hit covering that generated prefix, but both live quality checks returned repeated filler text. It is not a benchmark result. |
| Failed live handoff smoke | `348824841142825` | [`2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/`](2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/) | The PR #467 wheel was present, import probe and request metadata bridge passed, and SGLang reported a positive 128-token external cache hit, but the later 46-key split query still missed and both live quality checks failed. It is not a benchmark result. |
| Failed live handoff smoke | `672750124167579` | [`2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/`](2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/) | The PR #466 wheel was present, but the request metadata bridge failed during import probe because the strict hash-tracking gate looked for `get_hash_str` in the wrong SGLang controller lifecycle hook. It is not a benchmark result. |
| Failed live handoff smoke | `73938470896039` | [`2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/`](2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/) | The PR #465 wheel was present, but the later 46-key storage query still logged `last_hash_present=False` because SGLang keeps the per-batch prior hash in a local `_storage_hit_query` variable. It is not a benchmark result. |
| Failed live handoff smoke | `476430354490832` | [`2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/`](2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/) | The PR #464 provider wheel was present and hydrated the first 128 runtime pages, but Cachet still missed the later 46-key storage query because SGLang chained it from runtime `last_hash`. It is not a benchmark result. |
| Failed live handoff smoke | `521023980659718` | [`2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/`](2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/) | SGLang launched with `wait_complete` prefetch and reported 128 cached tokens from Cachet handoff pages, but Cachet missed the later 46-key storage query and both live checks failed. It is not a benchmark result. |
| Failed live handoff smoke | `201402713679607` | [`2026-06-23-g6-l4-live-handoff-smoke/`](2026-06-23-g6-l4-live-handoff-smoke/) | Generated handoff preparation completed, but SGLang rejected the colon-containing served-model name before live requests. It is not a benchmark result. |
| Failed live handoff smoke | `13763847664432` | [`2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/`](2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/) | SGLang launched with the safe served-model name and request metadata bridge validation passed, but the cache arm used suffix-only runtime prompt text and did not answer from the generated prefix. It is not a benchmark result. |
| Failed live handoff smoke | `476596508869043` | [`2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/`](2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/) | SGLang launched with logical prompt text and answered both arms, but the cache arm reported zero cached tokens; the later cached-token line was ordinary SGLang prefix-cache reuse. It is not a benchmark result. |
| Pending latency and throughput benchmark suite | Not available yet | Not available yet | The successful live smoke validates decode-time prefix binding and quality for one generated handoff prompt; Cachet still needs a multi-measurement SGLang benchmark suite before publishing latency or throughput numbers. |
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

Those helpers are a live readiness path, not a latency or throughput benchmark
suite. The current successful generated-handoff Databricks smoke is tracked in
[`2026-06-24-g6-l4-live-handoff-smoke-baseline-isolated-success/`](2026-06-24-g6-l4-live-handoff-smoke-baseline-isolated-success/).
Earlier generated-handoff attempts are tracked in
[`2026-06-23-g6-l4-live-handoff-smoke/`](2026-06-23-g6-l4-live-handoff-smoke/)
and
[`2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/`](2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/),
plus the zero-cache-hit blocker in
[`2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/`](2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/),
the partial page-binding blocker in
[`2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/`](2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/),
the chained hash-binding blocker in
[`2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/`](2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/),
the batch prior-hash metadata blocker in
[`2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/`](2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/),
the attach-time hash tracking gate in
[`2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/`](2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/),
the cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/`](2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/),
the token-stable cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-token-stable-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-token-stable-cache-hit-quality-failure/),
the Qwen-chat cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-qwen-chat-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-qwen-chat-cache-hit-quality-failure/),
the Qwen-sampling cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-qwen-sampling-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-qwen-sampling-cache-hit-quality-failure/),
the chat-completions cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-chat-completions-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-chat-completions-cache-hit-quality-failure/),
the no-thinking cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-no-thinking-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-no-thinking-cache-hit-quality-failure/),
the deterministic no-thinking cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-deterministic-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-deterministic-cache-hit-quality-failure/),
the Triton/PyTorch deterministic cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-triton-deterministic-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-triton-deterministic-cache-hit-quality-failure/),
the minimal no-thinking cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-minimal-no-thinking-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-minimal-no-thinking-cache-hit-quality-failure/),
the canary-after-cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-canary-after-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-canary-after-cache-hit-quality-failure/),
and the canary-flush cache-hit quality failure in
[`2026-06-24-g6-l4-live-handoff-smoke-canary-flush-cache-hit-quality-failure/`](2026-06-24-g6-l4-live-handoff-smoke-canary-flush-cache-hit-quality-failure/).
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
`get_hash_str`, not merely `operation.last_hash`. The latest run also carries
Qwen sampling parameters through the live request path. Recent runs show that
attach-time hash tracking installs successfully and that SGLang can report a
positive cache-arm external hit. Recent runs validate generated prefixes and
positive SGLang cached-token validation, including 175-token
chat-completions generated prefixes on g6/L4. Recent runs also
disables thinking through both `reasoning_effort=none` and
`chat_template_kwargs`, and set deterministic
`temperature=0.0`. Recent runs also force `--attention-backend triton`,
`--sampling-backend pytorch`, and `--enable-deterministic-inference`; the
backend controls are accepted and radix cache remains enabled. Recent runs use
a minimal no-thinking live request body without `top_p`, `top_k`, `min_p`, or
`presence_penalty`. The latest successful run executes the clean baseline
first, flushes `/flush_cache` before the cache arm so baseline quality remains
independent, and validates the cache-arm request itself by its prompt-token
total. It then flushes `/flush_cache` again before the model-quality canary,
which runs with zero cached tokens and produces `cachet-green`. Manual handoff
inputs remain supported when callers already have a validated SGLang handoff
plus `document_kv.sglang_hicache_page_keys` metadata. This report remains
pending for latency and throughput publication until a Databricks g6/L4 or
g5/A10G run writes a multi-measurement benchmark record using the same
benchmark schema as the vLLM report.

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
- The live prompt format is valid for Qwen3/SGLang and passes the baseline and
  cache-arm quality checks.
- The model-quality canary passes after the baseline and cache-arm live checks.
- The cached-token validation is matched to the cache-arm request, not to server
  warmup traffic or later baseline prefix-cache reuse.
- Latency, throughput, and quality measurements are recorded with the same
  benchmark schema used by the vLLM report.

## Evidence Boundary

The native probe source records under `../databricks/` prove that Cachet can
wire provider-backed dynamic HiCache integration in the target runtime. They
must not be cited as SGLang latency, throughput, or quality benchmark results.
Do not use `../../pr-evidence/` as the benchmark report surface.
