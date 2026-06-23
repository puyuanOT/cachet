# AWS g5/A10G V1 Compatibility Benchmark

This folder records the current non-default compatibility benchmark for Cachet.
It is useful for g5 cluster comparison, but it does not replace the strict
AWS g6/L4 release target.

| Field | Value |
| --- | --- |
| Databricks run | `566743786103032` |
| Run name | `Cachet vLLM hot g5 comparable payload cachet_vllm_hot_payload_g5_longcmp_388ea0a_20260623_162302_repeat3_cache8g_cachet_kv_current_main` |
| Hardware target | `aws-g5-a10g` |
| Node type | `g5.8xlarge` |
| Model | `qwen3:4b-instruct` |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Measurements | 24 |
| Compatibility role | `compatibility_benchmark` with matching `compatibility_databricks_run_status` |

## Results

| Dataset | TTFT speedup | Time-to-completion speedup | Answer found delta |
| --- | ---: | ---: | ---: |
| biography | 4.69x | 2.04x | 0.0 |
| hotpotqa | 6.04x | 2.67x | 0.0 |
| musique | 4.66x | 2.39x | 0.0 |
| niah | 5.91x | 2.66x | 0.0 |

## Artifacts

- [`v1_benchmark.json`](v1_benchmark.json) contains the
  `document_kv.benchmark_run.v1` compatibility benchmark report.
- [`databricks_run_status.json`](databricks_run_status.json) contains the
  terminal successful Databricks run-status summary for the compatibility run.
