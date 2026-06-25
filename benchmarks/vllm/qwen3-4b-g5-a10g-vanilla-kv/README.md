# vLLM Qwen3 4B On g5/A10G With Vanilla KV

This is Cachet's vLLM compatibility benchmark for g5/A10G hardware. It uses
the same model, method, baseline, and dataset suite as the primary g6/L4
result.

| Field | Value |
| --- | --- |
| Serving platform | vLLM |
| Model | `qwen3:4b-instruct` |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Method | Vanilla external KV cache |
| Baseline | `baseline_prefill` |
| Cache arm | `document_kv_cache` |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Measurements | 24 |
| Result | Cachet is faster than baseline with unchanged answer quality |

## Result

| Dataset | TTFT Speedup | Time-To-Completion Speedup | Answer Found Delta |
| --- | ---: | ---: | ---: |
| biography | 4.69x | 2.04x | 0.0 |
| hotpotqa | 6.04x | 2.67x | 0.0 |
| musique | 4.66x | 2.39x | 0.0 |
| niah | 5.91x | 2.66x | 0.0 |

## Scope

This result is useful for users running g5 clusters, but it does not replace
the primary g6/L4 target.

## Provenance

Sanitized evidence is committed beside this README:

- `v1_benchmark.json`
- `databricks_run_status.json`

The matching Databricks audit mirror is
[`../../databricks/vllm-qwen3-4b-g5-a10g-vanilla-kv/`](../../databricks/vllm-qwen3-4b-g5-a10g-vanilla-kv/).
