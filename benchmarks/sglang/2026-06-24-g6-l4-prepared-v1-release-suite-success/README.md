# SGLang g6/L4 Prepared V1 Release Suite Benchmark

This folder tracks the first successful Databricks prepared SGLang V1 live
benchmark run for Cachet. It is standalone benchmark evidence, not
`pr-evidence/` and not ignored local `databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `48413356233422` |
| Task run | `1003064866180856` |
| Run name | `Cachet SGLang prepared V1 sglang-prepared-v1-g6-5c66990-20260624_072154` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `5c66990` |
| Package wheel | `cachet_kv-0.2.0-py3-none-any.whl`, sha256 `f155fc10dfabae8ef8eb472075fc2d39496a7a97dc88a52aae11ffee60275c7e` |
| Mode | Prepared SGLang V1 benchmark datasets, generated SGLang handoff bundles, plain completion requests, two live benchmark repeats, `--attention-backend triton`, `--sampling-backend pytorch`, `--enable-deterministic-inference`, `hicache_storage_prefetch_policy=wait_complete`, page size `16`, `bfloat16` handoff generation |
| Current state | `SUCCESS`; import probe, request metadata bridge, prepared handoff generation, prepared handoff coverage, SGLang server launch, live measurement writing, and cache-hit validation all passed |
| Benchmark result | Prepared live V1 release-suite benchmark passed quality and cache-hit gates, but the Cachet cache arm was slower on these short prompts |

## Scope

This run wrote `cachet.sglang_live_benchmark.v1` with
`scope=live_v1_release`, `release_v1_suite=true`, Biography, HotpotQA, MusiQue,
and NIAH, one prepared example per dataset, and two baseline/cache repeats.

All 16 request measurements completed without request errors. All eight
Cachet cache-arm validations passed against generated handoff cached-token
floors. Baseline and Cachet answer-found quality both stayed at `1.0`, with
quality deltas of `0.0` for every dataset.

This result is intentionally not reported as a speedup. The Cachet cache arm
was slower than the baseline on all four short prepared prompts. It is still a
valid SGLang prepared live V1 benchmark result because the serving endpoint,
native HiCache provider, generated handoff path, cache-hit validation, and
quality gates all passed on the target AWS g6/L4 hardware.

## Result Snapshot

| Dataset | Baseline p50 TTFT | Cache p50 TTFT | TTFT speedup | Baseline p50 TTC | Cache p50 TTC | TTC speedup | Validated cached tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Biography | `0.197s` | `0.204s` | `0.966x` | `0.727s` | `0.753s` | `0.966x` | `96` |
| HotpotQA | `0.081s` | `0.257s` | `0.314x` | `1.412s` | `1.585s` | `0.891x` | `144` |
| MusiQue | `0.081s` | `0.225s` | `0.358x` | `1.416s` | `1.557s` | `0.910x` | `144` |
| NIAH | `0.077s` | `0.245s` | `0.313x` | `1.410s` | `1.576s` | `0.895x` | `96` |

## Artifacts

- [`success_run.json`](success_run.json) contains a compact, sanitized snapshot
  of the terminal successful Databricks state, import probe, prepared handoff
  generation and coverage, live benchmark rows, comparisons, and cache-hit
  validations.

Raw Databricks API responses, package wheels, driver logs, generated datasets,
handoff payloads, page-key lists, prompt text, task-output blobs, and local
scratch outputs stay out of this folder.
