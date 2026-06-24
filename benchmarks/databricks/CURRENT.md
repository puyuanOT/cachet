# Current Databricks Benchmark Snapshot

This is the human-readable Databricks artifact snapshot for the current Cachet
V1 benchmark evidence. For standalone benchmark reports, start with
[`../vllm/`](../vllm/), [`../sglang/`](../sglang/),
[`../storage/`](../storage/), and [`../native-engine/`](../native-engine/). This
snapshot summarizes the tracked Databricks results without requiring readers to
inspect `pr-evidence/` or ignored local `databricks-runs/` output.

| Field | Current value |
| --- | --- |
| Snapshot date | 2026-06-24 |
| Strict publication target | AWS g6/L4, `aws-g6-l4`, `g6.8xlarge` |
| Compatibility target | AWS g5/A10G, `aws-g5-a10g`, `g5.8xlarge` |
| Model | `qwen3:4b-instruct` |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Baseline arm | `baseline_prefill` |
| Cache arm | `document_kv_cache` |
| Release evidence | `ok=true` for the tracked g6/L4 benchmark, storage, and native probe artifacts |

## Scope

The full release latency and quality results in this snapshot are vLLM
benchmark runs. SGLang currently has provider-backed native HiCache probe and
connector-action evidence in this benchmark tree, plus a successful synthetic
live benchmark tracked at
[`../sglang/2026-06-24-g6-l4-live-benchmark-synthetic-niah-success/`](../sglang/2026-06-24-g6-l4-live-benchmark-synthetic-niah-success/)
and a successful generated-handoff live smoke tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-baseline-isolated-success/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-baseline-isolated-success/).
The latest prepared four-dataset SGLang V1 attempt is tracked at
[`../sglang/2026-06-24-g6-l4-prepared-v1-padded-token-validation-failure/`](../sglang/2026-06-24-g6-l4-prepared-v1-padded-token-validation-failure/):
run `918882025776007` on `aws-g6-l4` / `g6.8xlarge`, where import probe,
request metadata bridge, prepared handoff generation, prepared handoff
coverage, SGLang server launch, and live measurement writing succeeded. The
run wrote 16 `cachet.sglang_live_benchmark.v1` rows for Biography, HotpotQA,
MusiQue, and NIAH, but publication validation failed because SGLang logged
padded/page-rounded prompt-token totals for cache-arm prefill rows. The
previous config-swap failure is tracked at
[`../sglang/2026-06-24-g6-l4-prepared-v1-config-swap-failure/`](../sglang/2026-06-24-g6-l4-prepared-v1-config-swap-failure/):
run `514040136831626`, which generated all prepared handoffs but failed before
SGLang server launch because generated prepared datasets were combined with
single live handoff fields.
Earlier failed generated-handoff live smoke attempts remain tracked under
[`../sglang/2026-06-23-g6-l4-live-handoff-smoke/`](../sglang/2026-06-23-g6-l4-live-handoff-smoke/)
and
[`../sglang/2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/`](../sglang/2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/),
with the logical-prompt zero-cache-hit blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/),
the `wait_complete` partial page-binding blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/),
the chained `last_hash` page-binding blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/),
the batch prior-hash metadata blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/),
the attach-time hash tracking blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/),
the cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/),
the token-stable cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-token-stable-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-token-stable-cache-hit-quality-failure/),
the Qwen-chat cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-qwen-chat-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-qwen-chat-cache-hit-quality-failure/),
the Qwen-sampling cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-qwen-sampling-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-qwen-sampling-cache-hit-quality-failure/),
the chat-completions cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-chat-completions-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-chat-completions-cache-hit-quality-failure/),
the no-thinking cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-no-thinking-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-no-thinking-cache-hit-quality-failure/),
the deterministic no-thinking cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-deterministic-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-deterministic-cache-hit-quality-failure/),
the Triton/PyTorch deterministic cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-triton-deterministic-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-triton-deterministic-cache-hit-quality-failure/),
the minimal no-thinking cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-minimal-no-thinking-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-minimal-no-thinking-cache-hit-quality-failure/),
the canary-after-cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-canary-after-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-canary-after-cache-hit-quality-failure/),
and the canary-flush cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-canary-flush-cache-hit-quality-failure/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-canary-flush-cache-hit-quality-failure/).
The synthetic live benchmark validates decode-time prefix binding with Cachet
handoffs, hydrates matching generated page-key chunks, records positive
cache-arm cached-token validation covering the generated handoff prefix across
two repeats, keeps a post-flush model-quality canary, and preserves live
quality. It did not show a speedup on the tiny synthetic prompt. Treat full
SGLang latency and throughput evidence as pending until a multi-dataset SGLang
benchmark record covers the V1 release suite.

## V1 Latency And Quality

| Target | Folder | Databricks run | Measurements | TTFT speedup | Time-to-completion speedup | Quality delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Strict g6/L4 | [`2026-06-23-g6-l4-v1`](2026-06-23-g6-l4-v1/) | `872615985402004` | 24 | 5.27x-6.97x | 1.74x-2.25x | 0.0 |
| g5/A10G compatibility | [`2026-06-23-g5-a10g-v1-compatibility`](2026-06-23-g5-a10g-v1-compatibility/) | `566743786103032` | 24 | 4.66x-6.04x | 2.04x-2.67x | 0.0 |

The g5/A10G benchmark is compatibility evidence only. It is bundled through the
`compatibility_benchmark` and `compatibility_databricks_run_status` roles and
does not replace the strict AWS g6/L4 publication target.

## SGLang Synthetic Live Benchmark

| Target | Folder | Databricks run | Measurements | TTFT speedup | Time-to-completion speedup | Quality delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Synthetic g6/L4 | [`../sglang/2026-06-24-g6-l4-live-benchmark-synthetic-niah-success`](../sglang/2026-06-24-g6-l4-live-benchmark-synthetic-niah-success/) | `238535418152934` | 4 | 0.875x | 0.926x | 0.0 |

This SGLang artifact is scoped to one synthetic NIAH live prompt. It validates
two Cachet-backed cache repeats with 175 cached tokens each, but it does not
replace the full V1 release benchmark suite.

## SGLang Prepared V1 Attempts

| Target | Folder | Databricks run | Measurements | Result |
| --- | --- | --- | ---: | --- |
| Prepared g6/L4 | [`../sglang/2026-06-24-g6-l4-prepared-v1-padded-token-validation-failure`](../sglang/2026-06-24-g6-l4-prepared-v1-padded-token-validation-failure/) | `918882025776007` | 16 | Not published; cache-hit validation failed on padded SGLang prompt-token totals |
| Prepared g6/L4 | [`../sglang/2026-06-24-g6-l4-prepared-v1-config-swap-failure`](../sglang/2026-06-24-g6-l4-prepared-v1-config-swap-failure/) | `514040136831626` | 0 | Not published; failed before SGLang server launch |

## Storage And Native Engine Evidence

| Evidence | Folder | Databricks run | Target | Result |
| --- | --- | --- | --- | --- |
| Memory, Disk, and Unity Catalog storage readers | [`2026-06-21-g6-l4-storage-readers`](2026-06-21-g6-l4-storage-readers/) | `948365719597221` | `aws-g6-l4` / `g6.8xlarge` | Real UC Volume, zero reader errors |
| vLLM and SGLang provider-backed native probes | [`2026-06-23-g6-l4-native-engine-probes`](2026-06-23-g6-l4-native-engine-probes/) | `934698284395881` | `aws-g6-l4` / `g6.8xlarge` | Both backend tasks terminated `SUCCESS` |

The native probe folder carries `document_kv.engine_kv_connector_probe.v1` and
`document_kv.engine_kv_connector_actions.v1` records for both vLLM and SGLang.
Those records prove Cachet uses established engine-owned KV block managers
instead of a package-owned serving scheduler. They are not latency, throughput,
or quality benchmark measurements.

## Artifact Boundary

Tracked benchmark folders contain README summaries plus sanitized JSON records
needed to audit the claims. Keep these artifacts here:

- `document_kv.benchmark_run.v1`
- `cachet.sglang_live_benchmark.v1`
- `document_kv.storage_benchmark.v1`
- `document_kv.databricks_run_status.v1`
- `document_kv.engine_kv_connector_probe.v1`
- `document_kv.engine_kv_connector_actions.v1`
- `document_kv.release_evidence.v1`

Do not put Databricks tokens, raw Jobs API responses, package wheels, local run
logs, generated datasets, or strict release-bundle scratch directories in this
tree. Release-bundle manifests are regenerated from these benchmark artifacts
plus current governance, hygiene, wheel, preflight, and PR-evidence sidecars
before publication.
