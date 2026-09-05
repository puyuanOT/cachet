# SGLang g6/L4 Prepared V1 Attempt: Padded Token Validation Failure

This folder tracks the Databricks prepared SGLang V1 attempt that reached live
benchmark rows for all four V1 datasets. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `918882025776007` |
| Task run | `131444738323102` |
| Run name | `Cachet SGLang prepared V1 sglang-prepared-v1-g6-06a9c76-20260624_125434` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `06a9c76` |
| Package wheel | `cachet_kv-0.2.0-py3-none-any.whl`, sha256 `fe8a92440264473a621303b8afd60d6f741f9d43b94b389069ad12f5398ea348` |
| Mode | Prepared SGLang V1 benchmark datasets, generated SGLang handoff bundles, plain completion requests, two live benchmark repeats, `--attention-backend triton`, `--sampling-backend pytorch`, `--enable-deterministic-inference`, `hicache_storage_prefetch_policy=wait_complete`, page size `16`, `bfloat16` handoff generation |
| Current state | `FAILED`; import probe, request metadata bridge, prepared handoff generation, prepared handoff coverage, SGLang server launch, and live measurement writing succeeded, but cache-hit publication validation rejected the run |
| Benchmark result | Not published; 16 live measurement rows were written, but the run remains pre-publication evidence until padded SGLang prompt-token totals are handled and rerun |

## Scope

This run supersedes the earlier config-swap failure: the runner generated
prepared handoffs for Biography, HotpotQA, MusiQue, and NIAH, validated the
prepared handoff coverage, launched SGLang `0.5.10.post1` on an NVIDIA L4, and
wrote `cachet.sglang_live_benchmark.v1` with four datasets, two baseline
repeats, and two Cachet cache-arm repeats.

The live benchmark artifact contains 16 request measurements and four
baseline-vs-cache comparisons. All request rows completed without request
errors, and `answer_found_rate` stayed unchanged between the baseline and
Cachet arms.

The run is not a published SGLang benchmark result because all eight cache-arm
validation records failed to match the cache request by exact prompt-token
count in the SGLang server log. The server log rows did show the expected
cached-token floor, but SGLang reported page-rounded prompt totals:

| Dataset | Cache request prompt tokens | Required cached tokens | Observed SGLang prefill row |
| --- | ---: | ---: | --- |
| Biography | 120 | 96 | `96` cached + `32` new = `128` total |
| HotpotQA | 184 | 144 | `144` cached + `48` new = `192` total |
| MusiQue | 189 | 144 | `144` cached + `48` new = `192` total |
| NIAH | 132 | 96 | `96` cached + `48` new = `144` total |

The next local fix should let cache-hit validation fall back to the expected
minimum cached-token floor when the exact prompt-token count is missing because
SGLang padded or rounded the logged total. After that fix, the same g6/L4
prepared V1 target needs a fresh Databricks rerun before the SGLang result can
be cited as a benchmark.

## Measurement Snapshot

These measurements are useful diagnostics, but they are not publication
numbers because the benchmark gate failed.

| Dataset | TTFT speedup | Time-to-completion speedup | Answer-found delta | Exact-match delta |
| --- | ---: | ---: | ---: | ---: |
| Biography | `0.801x` | `0.914x` | `0.0` | `0.0` |
| HotpotQA | `0.356x` | `0.910x` | `0.0` | `0.0` |
| MusiQue | `0.341x` | `0.893x` | `0.0` | `0.0` |
| NIAH | `0.316x` | `0.895x` | `0.0` | `0.0` |

## Artifacts

- [`failed_run.json`](failed_run.json) contains a compact, sanitized snapshot
  of the terminal Databricks state, import probe, prepared handoff generation
  and coverage, live benchmark summary, measurement rows, comparisons, and
  cache-hit validation diagnosis.

Raw Databricks API responses, package wheels, driver logs, generated datasets,
handoff payloads, page-key lists, prompt text, task-output blobs, and local
scratch outputs stay out of this folder.
