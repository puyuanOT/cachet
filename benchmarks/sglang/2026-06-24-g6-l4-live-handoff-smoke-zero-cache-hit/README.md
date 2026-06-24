# SGLang g6/L4 Live Handoff Smoke: Zero Cache-Hit Blocker

This folder tracks the latest generated-handoff SGLang live smoke on the target
Databricks hardware. It is standalone benchmark-readiness evidence, not
`pr-evidence/` and not ignored local `databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `476596508869043` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `c774e0c` |
| Mode | Generated live Cachet handoff, SGLang HiCache page keys, logical prompt text |
| Current state | `FAILED`; the cache arm answered, but SGLang reported zero cached tokens for the cache-arm prefill |
| Benchmark result | Not published; no SGLang latency, throughput, or quality result was produced from this run |

## Scope

This run proves the previous served-model-name and suffix-only prompt blockers
were cleared. SGLang launched with `qwen3-4b-instruct`, the import probe passed,
the request metadata bridge passed, the smoke used `cache_prompt_text_mode=logical`,
and the live handoff generator produced 150 SGLang HiCache page keys for the
cached prefix.

The run still is not a benchmark result. The cache arm received Cachet
`kv_transfer_params` and answered the live check, but the SGLang server log
reported `#new-token: 174, #cached-token: 0` for that request. The later
`#cached-token: 173` line came from ordinary SGLang prefix-cache reuse after
the cache arm, not from the external Cachet handoff. A publishable SGLang
benchmark needs positive cached-token validation on the cache-arm request
itself.

## Promotion Criteria

Promote a replacement smoke run from readiness evidence to a published SGLang
benchmark only if all of these are true:

- A new Databricks run terminates with `result_state=SUCCESS` on `g6.8xlarge`.
- The SGLang live smoke writes `sglang-live-smoke.json` with `ok=true`.
- The cache arm uses logical prompt text or an equivalent validated virtual
  prefix binding path.
- The cache-arm request reports positive SGLang cached-token validation.
- The positive cached-token validation is matched to the cache request's
  prompt-token total, not to warmup requests or later baseline prefix-cache
  reuse.
- Latency, throughput, and quality numbers are recorded in the same benchmark
  schema used by the vLLM report.

## Artifacts

- [`failed_run.json`](failed_run.json) contains a compact, sanitized snapshot
  of the terminal failed Databricks run state and blocker.

Raw Databricks API responses, package wheels, driver logs, generated payloads,
and local scratch outputs stay out of this folder.
