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

The latency and quality results in this snapshot are vLLM benchmark runs. SGLang
currently has provider-backed native HiCache probe and connector-action evidence
in this benchmark tree, plus failed generated-handoff live smoke attempts
tracked under
[`../sglang/2026-06-23-g6-l4-live-handoff-smoke/`](../sglang/2026-06-23-g6-l4-live-handoff-smoke/)
and
[`../sglang/2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/`](../sglang/2026-06-23-g6-l4-live-handoff-smoke-runtime-suffix/),
with the latest logical-prompt zero-cache-hit blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-zero-cache-hit/),
the `wait_complete` partial page-binding blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-partial-page-binding/),
the chained `last_hash` page-binding blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-chained-hash-binding/),
the batch prior-hash metadata blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-batch-prior-metadata/),
the attach-time hash tracking blocker tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-attach-hash-tracking/),
and the current cache-hit quality failure tracked at
[`../sglang/2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/`](../sglang/2026-06-24-g6-l4-live-handoff-smoke-quality-failure-cache-hit/).
It does not yet have a published live SGLang latency/quality benchmark. Treat
SGLang benchmark evidence as pending until a live SGLang endpoint validates
decode-time prefix binding with Cachet handoffs, hydrates all matching
generated page-key chunks, records a positive cache-arm cached-token
validation, and passes the live quality gate.

## V1 Latency And Quality

| Target | Folder | Databricks run | Measurements | TTFT speedup | Time-to-completion speedup | Quality delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Strict g6/L4 | [`2026-06-23-g6-l4-v1`](2026-06-23-g6-l4-v1/) | `872615985402004` | 24 | 5.27x-6.97x | 1.74x-2.25x | 0.0 |
| g5/A10G compatibility | [`2026-06-23-g5-a10g-v1-compatibility`](2026-06-23-g5-a10g-v1-compatibility/) | `566743786103032` | 24 | 4.66x-6.04x | 2.04x-2.67x | 0.0 |

The g5/A10G benchmark is compatibility evidence only. It is bundled through the
`compatibility_benchmark` and `compatibility_databricks_run_status` roles and
does not replace the strict AWS g6/L4 publication target.

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
