# SGLang g6/L4 Live Handoff Smoke: Chat Completions Cache Hit With Quality Failure

This folder tracks the generated-handoff SGLang live smoke launched after PR
#473 merged on the target Databricks hardware. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `163920824964705` |
| Task run | `412030078319178` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `ff868a4` |
| Mode | Generated live Cachet handoff, logical prompt text, Qwen3 chat-format live checks through `/v1/chat/completions`, Qwen sampling, tokenizer-rendered handoff prompt, token-stable full-page prefix generation, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete` |
| Current state | `FAILED`; import probe, request metadata bridge, generated Qwen-chat live handoff, chat-completions request plumbing, and cache-hit validation passed; both live quality checks failed |
| Benchmark result | Not published; no SGLang latency, throughput, or quality benchmark result is claimed from this run |

## Scope

This run proves that the PR #473 chat-completions request path reached the live
SGLang runtime on AWS g6/L4. The generated live handoff used the tokenizer chat
template, produced a 175-token full-page prefix after truncating a 176-token
source prefix to a token-stable runtime prefix, and attached the matching
SGLang HiCache page-key metadata. Cachet hydrated all 175 generated prefix
pages, and SGLang reported 175 cached tokens for the cache-arm request out of
205 total prompt tokens.

The run is still not publishable. Both the baseline and cache arm returned
repeated filler text instead of the expected NIAH answer through the
`/v1/chat/completions` path. The next SGLang serving fix should isolate whether
Qwen3/SGLang needs a different chat template, chat request body, generation
settings, or tokenizer-rendered completion prompt before promotion.

## Promotion Criteria

Promote a replacement smoke run from readiness evidence to a published SGLang
benchmark only if all of these are true:

- A new Databricks run terminates with `result_state=SUCCESS` on `g6.8xlarge`.
- The import probe reports `document_kv_request_metadata_bridge_ok=true`.
- The SGLang live smoke writes `sglang-live-smoke.json` with `ok=true`.
- Generated handoff creation records a token-stable, full-page cache prefix.
- The cache-arm request reports positive SGLang cached-token validation that
  covers the generated handoff prefix length.
- The live quality gate passes for both the baseline and cache arms.
- Latency, throughput, and quality numbers are recorded in the same benchmark
  schema used by the vLLM report.

## Artifacts

- [`failed_run.json`](failed_run.json) contains a compact, sanitized snapshot
  of the terminal failed Databricks run state and blocker.

Raw Databricks API responses, package wheels, driver logs, generated payloads,
page-key lists, and local scratch outputs stay out of this folder.
