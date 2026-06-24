# SGLang g6/L4 Live Synthetic NIAH Benchmark

This folder tracks the first SGLang run that wrote the opt-in
`cachet.sglang_live_benchmark.v1` artifact. It is standalone benchmark
evidence, not `pr-evidence/` and not ignored local `databricks-runs/` scratch
output.

| Field | Value |
| --- | --- |
| Databricks run | `238535418152934` |
| Task run | `760304813051328` |
| Run name | `document-kv-sglang-smoke` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Spark image | `15.4.x-gpu-ml-scala2.12` |
| Task | `document_kv_sglang_smoke` |
| Cachet commit | `eff801d` |
| Package wheel | `cachet_kv-0.2.0-py3-none-any.whl`, sha256 `c30e1cc64ae58796df4b2973dbef78d6ad708e8168133fa28d4acb0ea4e2fa4e` |
| Mode | Generated live Cachet handoff, clean baseline first, `/flush_cache` before the smoke cache arm, logical prompt text, Qwen3 chat-format checks, no-thinking request body, `temperature=0.0`, `--attention-backend triton`, `--sampling-backend pytorch`, `--enable-deterministic-inference`, `prefetch_threshold=1`, `hicache_storage_prefetch_policy=wait_complete`, `/flush_cache` before the model-quality canary, then two live benchmark repeats |
| Current state | `SUCCESS`; import probe, request metadata bridge, generated live handoff, smoke quality, smoke cache-hit validation, canary, and repeated live benchmark validations all passed |
| Benchmark result | Synthetic live benchmark passed quality and cache-hit gates, but the cache arm was slower on this tiny prompt: TTFT speedup `0.875x`; time-to-completion speedup `0.926x` |

## Scope

This run validates repeated SGLang live Cachet measurements on the target AWS
g6/L4 Databricks hardware. The runner wrote `sglang-live-benchmark.json` after
the live smoke passed. The benchmark scope is one synthetic NIAH prompt with
two baseline repeats and two Cachet cache-arm repeats.

Both Cachet cache-arm repeats validated 175 cached tokens, matching the
generated handoff prefix length. Baseline and cache-arm answer quality both
passed with answer-found rate `1.0`, and quality deltas were `0.0`.

This result is intentionally not reported as a speedup. On this small prompt,
the baseline p50 TTFT was `0.3029023165s`, while the Cachet cache-arm p50 TTFT
was `0.3463493005s`. The baseline p50 time-to-completion was `0.5560753460s`,
while the Cachet cache-arm p50 time-to-completion was `0.6006555025s`.

This is also not the full SGLang release benchmark suite. It proves the
Databricks g6/L4 SGLang path can now emit repeated live latency and quality
measurements with validated Cachet-backed external cache hits. The remaining
release benchmark work is the full four-dataset SGLang suite.

## Artifacts

- [`success_run.json`](success_run.json) contains a compact, sanitized snapshot
  of the terminal successful Databricks run state, smoke gates, live benchmark
  rows, comparisons, and cache-hit validations.

Raw Databricks API responses, package wheels, driver logs, generated payloads,
page-key lists, prompt text, and local scratch outputs stay out of this folder.
