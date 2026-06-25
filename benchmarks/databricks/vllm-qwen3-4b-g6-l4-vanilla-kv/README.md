# AWS g6/L4 V1 Benchmark

This folder records the current strict V1 benchmark evidence for Cachet on the
target Databricks hardware profile.

| Field | Value |
| --- | --- |
| Databricks run | `872615985402004` |
| Run name | `Cachet vLLM hot comparable payload cachet_vllm_hot_payload_longcmp_388ea0a_20260623_160711_repeat3_cache8g_cachet_kv_current_main` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Model | `qwen3:4b-instruct` |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Measurements | 24 |
| Release evidence | `ok=true`, no issues |

## Results

| Dataset | TTFT speedup | Time-to-completion speedup | Answer found delta |
| --- | ---: | ---: | ---: |
| biography | 5.27x | 1.74x | 0.0 |
| hotpotqa | 6.97x | 2.23x | 0.0 |
| musique | 5.33x | 2.06x | 0.0 |
| niah | 6.90x | 2.25x | 0.0 |

## Artifacts

- [`v1_benchmark.json`](v1_benchmark.json) contains the
  `document_kv.benchmark_run.v1` benchmark report.
- [`databricks_run_status.json`](databricks_run_status.json) contains the
  terminal successful Databricks run-status summary.
- [`release_evidence.json`](release_evidence.json) contains the strict
  release-evidence validation over the target benchmark, storage benchmark, and
  native vLLM/SGLang probe/action sidecars.
