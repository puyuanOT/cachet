# AWS g6/L4 V1 Benchmark

This standalone report records the current strict Cachet vLLM V1 benchmark on
the target Databricks hardware profile.

| Field | Value |
| --- | --- |
| Date | 2026-06-23 |
| Scope | vLLM V1 latency and quality |
| Target | `aws-g6-l4` / `g6.8xlarge` |
| Databricks run | `872615985402004` |
| Model | `qwen3:4b-instruct` |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Measurements | 24 |
| Result | Published benchmark; release evidence `ok=true` |

## Human Result

Cachet's `document_kv_cache` arm beat `baseline_prefill` on all four V1
datasets with unchanged answer quality. TTFT speedups ranged from 5.27x to
6.97x, and time-to-completion speedups ranged from 1.74x to 2.25x.

| Dataset | TTFT speedup | Time-to-completion speedup | Answer found delta |
| --- | ---: | ---: | ---: |
| biography | 5.27x | 1.74x | 0.0 |
| hotpotqa | 6.97x | 2.23x | 0.0 |
| musique | 5.33x | 2.06x | 0.0 |
| niah | 6.90x | 2.25x | 0.0 |

## Scope

This is the strict publication target: AWS g6/L4 on plain `g6.8xlarge`
Databricks hardware. It is a vLLM benchmark result, not an SGLang benchmark
result.

## Source Artifacts

The sanitized source records live in
[`../../databricks/2026-06-23-g6-l4-v1/`](../../databricks/2026-06-23-g6-l4-v1/):

- `v1_benchmark.json`
- `databricks_run_status.json`
- `release_evidence.json`

## Artifact Boundary

This folder is the human-readable benchmark report. Keep raw Databricks Jobs
API responses, tokens, wheels, logs, generated datasets, and local scratch
output out of this tree.
