# SGLang g6/L4 Live Handoff Smoke: Baseline-Isolated Success

This folder tracks the first generated-handoff SGLang live smoke that passed
after isolating the baseline from the Cachet cache arm. It is standalone
benchmark-readiness evidence, not `pr-evidence/` and not ignored local
`databricks-runs/` scratch output.

| Field | Value |
| --- | --- |
| Databricks run | `134006212072875` |
| Task run | `342507441509485` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `b33b3f4` |
| Package wheel | `cachet_kv-0.2.0-py3-none-any.whl`, sha256 `10ca1684238f40e437a0a1b565cdfa5e0ecdb105169ad3d704c98a56f9cb9053` |
| Mode | Generated live Cachet handoff, clean baseline first, `/flush_cache` before the cache arm, logical prompt text, Qwen3 chat-format live checks, minimal no-thinking request body, `temperature=0.0`, `--attention-backend triton`, `--sampling-backend pytorch`, `--enable-deterministic-inference`, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete`, and `/flush_cache` before the model-quality canary |
| Current state | `SUCCESS`; import probe, request metadata bridge, generated Qwen-chat live handoff, baseline quality, Cachet cache-arm quality, full external cache-hit validation, and post-flush canary all passed |
| Benchmark result | Readiness pass; not yet a latency or throughput benchmark suite |

## Scope

This run validates the SGLang live Cachet path on the target AWS g6/L4
Databricks hardware. The smoke ran a clean baseline first, flushed SGLang's
ordinary prefix cache, then ran the Cachet handoff-backed cache arm. The cache
arm reported 175 cached tokens out of a 205-token prompt, matching the
generated handoff prefix length, while the baseline and cache-arm outputs both
returned `otkv7391`.

The smoke also flushed SGLang's prefix cache before the model-quality canary.
The canary then ran with zero cached tokens and produced `cachet-green`, so the
previous quality-contamination signal is cleared for this smoke path.

This is still not a published SGLang latency/throughput benchmark. It is a
single live handoff smoke that proves Cachet can bind generated SGLang HiCache
pages into the runtime, validate a positive cache-arm hit, and preserve the
quality gates on `g6.8xlarge`. The next benchmark step is to promote this path
into a multi-measurement SGLang latency and throughput suite using the same
schema as the published vLLM report.

## Artifacts

- [`success_run.json`](success_run.json) contains a compact, sanitized snapshot
  of the terminal successful Databricks run state and smoke evidence.

Raw Databricks API responses, package wheels, driver logs, generated payloads,
page-key lists, prompt text, and local scratch outputs stay out of this folder.
