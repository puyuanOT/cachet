# SGLang g6/L4 Live Handoff Smoke: Deterministic Cache Hit With Quality Failure

This folder tracks the generated-handoff SGLang live smoke launched after PR
#478 merged on the target Databricks hardware. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `647563677081667` |
| Task run | `522972650970006` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `b89cad5` |
| Mode | Generated live Cachet handoff, logical prompt text, Qwen3 chat-format live checks through `/v1/chat/completions`, deterministic `temperature=0.0`, no-thinking request controls, tokenizer-rendered handoff prompt, token-stable full-page prefix generation, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete` |
| Current state | `FAILED`; import probe, request metadata bridge, generated Qwen-chat live handoff, deterministic no-thinking chat request/template metadata, and cache-hit validation passed; both live quality checks failed |
| Benchmark result | Not published; no SGLang latency, throughput, or quality benchmark result is claimed from this run |

## Scope

This run proves that the PR #478 wheel reached the live SGLang runtime on AWS
g6/L4 with deterministic live-check decoding. The Databricks submit payload and
the smoke metadata both recorded `live_check_temperature=0.0`, while preserving
`reasoning_effort=none`, `thinking=false`, and `enable_thinking=false` for the
Qwen3 chat-template path.

The generated live handoff used the tokenizer chat template, produced a
175-token full-page prefix after truncating a 176-token source prefix to a
token-stable runtime prefix, and attached the matching SGLang HiCache page-key
metadata. Cachet hydrated all 175 generated prefix pages, and SGLang reported 175 cached
tokens for the cache-arm request out of 205 total prompt tokens.

The live smoke still is not publishable. Both the baseline and cache arm
returned the same repeated filler text instead of the expected NIAH answer
through the `/v1/chat/completions` path. Because deterministic decoding did
not change the quality failure, the next SGLang serving fix should isolate why
the Qwen3/SGLang chat endpoint produces filler text even with deterministic
sampling and no-thinking chat-template controls.

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
