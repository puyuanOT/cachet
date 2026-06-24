# SGLang g6/L4 Live Handoff Smoke: Runtime Suffix Blocker

This folder tracks the second generated-handoff SGLang live smoke on the target
Databricks hardware. It is standalone benchmark-readiness evidence, not
`pr-evidence/` and not ignored local `databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `13763847664432` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `425ee36` |
| Mode | Generated live Cachet handoff plus SGLang HiCache page keys |
| Current state | `FAILED`; cache arm used suffix-only runtime prompt text and did not answer from the generated prefix |
| Benchmark result | Not published; no SGLang latency, throughput, or quality result was produced from this run |

## Scope

This run proves the previous served-model-name blocker was cleared. SGLang
launched with `qwen3-4b-instruct`, the import probe passed, the request metadata
bridge passed, and the live handoff generator produced 150 SGLang HiCache page
keys for the cached prefix.

The run still is not a benchmark result. The cache arm used
`cache_prompt_text_mode=runtime`, which sends only the runtime suffix to stock
SGLang. That path does not give SGLang the logical prefix tokens it needs to
compute the cached prefix page keys. The baseline answered the live check, but
the cache arm did not.

## Promotion Criteria

Promote a replacement smoke run from readiness evidence to a published SGLang
benchmark only if all of these are true:

- A new Databricks run terminates with `result_state=SUCCESS` on `g6.8xlarge`.
- The SGLang live smoke writes `sglang-live-smoke.json` with `ok=true`.
- The cache arm uses logical prompt text or an equivalent validated virtual
  prefix binding path.
- The smoke records a positive SGLang cache-hit validation for the cache-arm
  request, not only a correct answer.
- Latency, throughput, and quality numbers are recorded in the same benchmark
  schema used by the vLLM report.

## Artifacts

- [`failed_run.json`](failed_run.json) contains a compact, sanitized snapshot
  of the terminal failed Databricks run state and blocker.

Raw Databricks API responses, package wheels, driver logs, generated payloads,
and local scratch outputs stay out of this folder.
