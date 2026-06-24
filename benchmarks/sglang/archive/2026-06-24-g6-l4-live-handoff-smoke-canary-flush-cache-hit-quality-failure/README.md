# SGLang g6/L4 Live Handoff Smoke: Canary Flush With Cache Hit And Quality Failure

This folder tracks the generated-handoff SGLang live smoke launched after PR
#488 merged on the target Databricks hardware. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `655273897262076` |
| Task run | `541340270733865` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `f95d4f0` |
| Package wheel | `cachet_kv-0.2.0-py3-none-any.whl`, sha256 `43185087ca4cd8148102bd7b813dded530ec9cd0e882b7496641ad9acc952c48` |
| Mode | Generated live Cachet handoff, logical prompt text, Qwen3 chat-format live checks, minimal no-thinking request body, `temperature=0.0`, `--attention-backend triton`, `--sampling-backend pytorch`, `--enable-deterministic-inference`, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete`, and `/flush_cache` before the model-quality canary |
| Current state | `FAILED`; import probe and request metadata bridge passed, generated Qwen-chat live handoff passed, SGLang hydrated the generated prefix, the cache arm reported a full 175-token cache hit, `/flush_cache` returned HTTP 200, and the post-flush model-quality canary passed, but baseline and cache-arm quality failed |
| Benchmark result | Not published; no SGLang latency, throughput, or quality benchmark result is claimed from this run |

## Scope

This run proves that PR #488 removed the previous post-live-check canary
contamination signal. SGLang accepted `/flush_cache?timeout=30`, logged a
successful cache flush, then ran the model-quality canary with zero cached
tokens and produced `cachet-green`.

The Cachet handoff path still worked in the same run. The generated Qwen-chat
handoff had 175 full pages, Cachet hydrated the expected SGLang HiCache page
keys, and the cache-arm request reported 175 cached tokens out of 205 prompt
tokens. That exactly covers the generated Cachet handoff prefix.

The run is still not publishable. Both the baseline and cache arms returned
the same repeated filler text instead of the expected NIAH answer. The current
blocker is therefore Qwen3/SGLang live request quality for the benchmark prompt,
not Cachet handoff hydration, request metadata bridging, cached-token
validation, or the post-live-check canary.

## Promotion Criteria

Promote a replacement smoke run from readiness evidence to a published SGLang
benchmark only if all of these are true:

- A new Databricks run terminates with `result_state=SUCCESS` on `g6.8xlarge`.
- The import probe reports `document_kv_request_metadata_bridge_ok=true`.
- The SGLang live smoke writes `sglang-live-smoke.json` with `ok=true`.
- Generated handoff creation records a full-page cache prefix.
- The cache-arm request reports positive SGLang cached-token validation that
  covers the generated handoff prefix length.
- The baseline, cache arm, and model-quality canary all pass quality checks.
- Latency, throughput, and quality numbers are recorded in the same benchmark
  schema used by the vLLM report.

## Artifacts

- [`failed_run.json`](failed_run.json) contains a compact, sanitized snapshot
  of the terminal failed Databricks run state and blocker.

Raw Databricks API responses, package wheels, driver logs, generated payloads,
page-key lists, prompt text, and local scratch outputs stay out of this folder.
