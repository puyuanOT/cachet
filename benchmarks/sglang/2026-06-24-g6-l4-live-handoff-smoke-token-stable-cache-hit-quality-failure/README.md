# SGLang g6/L4 Live Handoff Smoke: Token-Stable Cache Hit With Quality Failure

This folder tracks the generated-handoff SGLang live smoke launched after PR
#469 merged on the target Databricks hardware. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `995284076545208` |
| Task run | `815981906697847` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `d6c9660` |
| Mode | Generated live Cachet handoff, logical prompt text, token-stable full-page prefix generation, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete` |
| Current state | `FAILED`; import probe, request metadata bridge, and generated live handoff passed; SGLang reported a positive cache-arm hit for the full 149-token generated prefix, but both live quality checks failed |
| Benchmark result | Not published; no SGLang latency, throughput, or quality benchmark result is claimed from this run |

## Scope

This run proves that the PR #469 token-stable handoff fix reached the live
SGLang runtime. Generated live handoff creation produced a 149-token full-page
prefix after truncating a 150-token source prefix to a token-stable runtime
prefix. The cache-arm prefill then reported 149 cached tokens out of 174 prompt
tokens, and Cachet's cache-hit validation accepted the hit because it covered
the generated handoff prefix length.

The run is still not publishable. Both the baseline and cache arm returned a
repeated filler token instead of the expected NIAH answer, so the live quality
gate failed even though external-prefix hydration was validated. The next
SGLang serving fix must make the live Qwen3 prompt format reliable before
promotion.

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
