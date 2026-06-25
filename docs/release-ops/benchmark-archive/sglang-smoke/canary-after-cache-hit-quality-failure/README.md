# SGLang g6/L4 Live Handoff Smoke: Canary After Cache Hit With Quality Failure

This folder tracks the generated-handoff SGLang live smoke launched after PR
#486 merged on the target Databricks hardware. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `419314952937106` |
| Task run | `42608288205489` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `16a8085` |
| Package wheel | `cachet_kv-0.2.0-py3-none-any.whl`, sha256 `de79f5c8b91b333c4279b47ee85676f42ead3f4db4a317820632216384a93d5c` |
| Mode | Generated live Cachet handoff, logical prompt text, Qwen3 chat-format live checks, minimal no-thinking request body, `temperature=0.0`, `--attention-backend triton`, `--sampling-backend pytorch`, `--enable-deterministic-inference`, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete`, and model-quality canary after cache/baseline checks |
| Current state | `FAILED`; import probe and request metadata bridge passed, generated Qwen-chat live handoff passed, SGLang hydrated the generated prefix, and the cache arm reported a full 175-token cache hit, but the baseline, cache arm, and post-live-check model-quality canary all failed quality |
| Benchmark result | Not published; no SGLang latency, throughput, or quality benchmark result is claimed from this run |

## Scope

This run proves that PR #486 fixed the prior ordering problem where the
model-quality canary perturbed the cache-arm validation. The canary now runs
after the cache and baseline live checks, and the cache-arm request reported
175 cached tokens out of 205 prompt tokens. That exactly covers the generated
175-token Cachet handoff prefix.

The runtime integration checks also passed. SGLang loaded Cachet's dynamic
HiCache provider, the request metadata bridge passed, CUDA reported an
`NVIDIA L4`, and the live request shape used `/v1/chat/completions` with
`max_completion_tokens`, Qwen chat messages, the minimal no-thinking body, and
SGLang `custom_params` carrying Cachet handoff metadata.

The run is still not publishable. Both the baseline and cache arms returned
the same repeated filler text instead of the expected NIAH answer, and the
post-live-check model-quality canary failed to produce `cachet-green`. A prior
diagnostic run showed the same canary can pass when it runs before the live
cache/baseline checks, so the remaining blocker appears to be SGLang/Qwen
generation quality or prefix-cache state/order after the live checks, not
Cachet handoff hydration or request-shape plumbing.

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
