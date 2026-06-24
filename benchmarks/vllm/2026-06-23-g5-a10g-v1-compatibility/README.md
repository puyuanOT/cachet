# AWS g5/A10G V1 Compatibility Benchmark

This standalone report records the current non-default Cachet vLLM V1
compatibility benchmark on Databricks g5/A10G hardware.

| Field | Value |
| --- | --- |
| Date | 2026-06-23 |
| Scope | vLLM V1 latency and quality compatibility |
| Target | `aws-g5-a10g` / `g5.8xlarge` |
| Databricks run | `566743786103032` |
| Model | `qwen3:4b-instruct` |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Measurements | 24 |
| Result | Compatibility evidence; not the strict release target |

## Human Result

Cachet's `document_kv_cache` arm beat `baseline_prefill` on all four V1
datasets with unchanged answer quality. TTFT speedups ranged from 4.66x to
6.04x, and time-to-completion speedups ranged from 2.04x to 2.67x.

| Dataset | TTFT speedup | Time-to-completion speedup | Answer found delta |
| --- | ---: | ---: | ---: |
| biography | 4.69x | 2.04x | 0.0 |
| hotpotqa | 6.04x | 2.67x | 0.0 |
| musique | 4.66x | 2.39x | 0.0 |
| niah | 5.91x | 2.66x | 0.0 |

## Scope

This g5/A10G run is useful compatibility evidence for g5 clusters, but it does
not replace the strict AWS g6/L4 publication target.

## Source Artifacts

The sanitized source records are committed beside this README:

- `v1_benchmark.json`
- `databricks_run_status.json`

The same records are mirrored under
[`../../databricks/2026-06-23-g5-a10g-v1-compatibility/`](../../databricks/2026-06-23-g5-a10g-v1-compatibility/)
for release-bundle and Databricks run-status audits.

## Artifact Boundary

This folder is the human-readable benchmark report. Keep raw Databricks Jobs
API responses, tokens, wheels, logs, generated datasets, and local scratch
output out of this tree.
