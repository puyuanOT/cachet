# SGLang g6/L4 Live Handoff Smoke

This folder tracks the current human-readable SGLang live handoff smoke run on
the target Databricks hardware. It is a standalone benchmark-readiness folder,
not `pr-evidence/` and not ignored local `databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `201402713679607` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `86a8085` |
| Mode | Generated live Cachet handoff plus SGLang HiCache page keys |
| Current state | `FAILED`; server rejected the colon-containing SGLang served-model name before live requests |
| Benchmark result | Not published; no SGLang latency, throughput, or quality result was produced from this run |

## Scope

This run exercises the generated live handoff path added for SGLang. The job
creates the synthetic Cachet handoff and the matching SGLang HiCache page-key
metadata inside the target runtime before starting the SGLang server.

Do not cite this folder as a completed benchmark. The Databricks run reached a
terminal failed state before live requests could run. The source of truth is the
sanitized failure summary in [`failed_run.json`](failed_run.json).

The useful signal from this attempt is that generated live handoff preparation
completed and the SGLang server launch reached argument validation. SGLang
0.5.10 then rejected `qwen3:4b-instruct` as `--served-model-name` because the
colon is reserved for `model:adapter` syntax.

## Promotion Criteria

Promote a replacement smoke run from readiness evidence to a published SGLang
benchmark only if all of these are true:

- A new Databricks run terminates with `result_state=SUCCESS` on `g6.8xlarge`.
- The SGLang live smoke writes `sglang-live-smoke.json` with `ok=true`.
- The handoff-backed cache arm validates decode-time prefix binding against the
  live SGLang endpoint.
- Any latency, throughput, and quality numbers are recorded in the same
  benchmark schema used by the vLLM report.

## Artifacts

- [`failed_run.json`](failed_run.json) contains a compact, sanitized snapshot
  of the terminal failed Databricks run state and blocker.

Raw Databricks API responses, package wheels, driver logs, generated payloads,
and local scratch outputs stay out of this folder.
