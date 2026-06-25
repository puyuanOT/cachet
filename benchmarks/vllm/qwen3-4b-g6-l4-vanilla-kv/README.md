# vLLM Qwen3 4B On g6/L4 With Vanilla KV

This is the primary Cachet vLLM benchmark result. It compares vLLM no-cache
prefill with Cachet's vanilla external KV cache arm on the target g6/L4
hardware.

| Field | Value |
| --- | --- |
| Serving platform | vLLM |
| Model | `qwen3:4b-instruct` |
| Hardware | AWS g6/L4, `g6.8xlarge` |
| Method | Vanilla external KV cache |
| Baseline | `baseline_prefill` |
| Cache arm | `document_kv_cache` |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Measurements | 24 |
| Result | Cachet is faster than baseline with unchanged answer quality |

## Result

| Dataset | TTFT Speedup | Time-To-Completion Speedup | Answer Found Delta |
| --- | ---: | ---: | ---: |
| biography | 5.27x | 1.74x | 0.0 |
| hotpotqa | 6.97x | 2.23x | 0.0 |
| musique | 5.33x | 2.06x | 0.0 |
| niah | 6.90x | 2.25x | 0.0 |

## Provenance

Sanitized evidence is committed beside this README:

- `v1_benchmark.json`
- `databricks_run_status.json`
- `release_evidence.json`

The matching Databricks audit mirror is
[`../../databricks/vllm-qwen3-4b-g6-l4-vanilla-kv/`](../../databricks/vllm-qwen3-4b-g6-l4-vanilla-kv/).
Raw Databricks responses, credentials, wheels, logs, generated datasets, and
local scratch output are intentionally excluded.
