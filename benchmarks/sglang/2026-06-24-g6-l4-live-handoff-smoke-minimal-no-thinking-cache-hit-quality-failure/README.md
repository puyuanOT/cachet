# SGLang g6/L4 Live Handoff Smoke: Minimal No-Thinking Cache Hit With Quality Failure

This folder tracks the generated-handoff SGLang live smoke launched after PR
#482 merged on the target Databricks hardware. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `672927118707206` |
| Task run | `540954902597909` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `046ab03` |
| Package wheel | `cachet_kv-0.2.0-py3-none-any.whl`, sha256 `c355cb546b3818cfd31ae155479cc9412f6f06d81fbdb1d52e3d4c1deabd50a1` |
| Mode | Generated live Cachet handoff, logical prompt text, Qwen3 chat-format live checks, minimal no-thinking request body, `temperature=0.0`, `--attention-backend triton`, `--sampling-backend pytorch`, `--enable-deterministic-inference`, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete` |
| Current state | `FAILED`; import probe and request metadata bridge passed, generated Qwen-chat live handoff passed, the minimal no-thinking request body reached the live request path, and SGLang reported a positive 175-token cache-arm hit, but both live quality checks failed |
| Benchmark result | Not published; no SGLang latency, throughput, or quality benchmark result is claimed from this run |

## Scope

This run proves that PR #482's minimal SGLang live-check request body reached
the Databricks g6/L4 runtime. The live request body contained only
`reasoning_effort=none` plus `chat_template_kwargs` with `thinking=false` and
`enable_thinking=false`; the prior `top_p`, `top_k`, `min_p`, and
`presence_penalty` sampling extras were absent. SGLang launched with
`attention_backend='triton'`, `sampling_backend='pytorch'`, and
`enable_deterministic_inference=True` while leaving radix cache enabled.

Cachet's request metadata bridge passed, the generated live handoff produced a
175-token full-page prefix after truncating the 176-token source prefix, and
the cache-arm request reported 175 cached tokens out of 205 prompt tokens.

The run is still not publishable. Both the baseline and cache arms returned
the same repeated filler text instead of the expected NIAH answer:
`or or or Cover Cover Cover ...`. This rules out the removed sampling extras as
the cause of the repeated-filler failure. The current blocker remains the
Qwen3/SGLang live quality path on L4, not backend flag plumbing, request
metadata transport, generated handoff creation, or Cachet HiCache hydration.

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
