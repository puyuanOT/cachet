# vLLM Qwen3 4B On g5/A10G With Vanilla KV

Appendix evidence for an existing vLLM g5/A10G compatibility benchmark. It is
not a row in the primary table in [benchmark root](../../../).

## Experimental Setup

| Field | Value |
| --- | --- |
| Engine | vLLM |
| Model | `qwen3:4b-instruct` |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Method | Vanilla external KV |
| Baseline arm | `baseline_prefill` |
| Cache arm | `document_kv_cache` |
| Dataset scope | Biography, HotpotQA, MusiQue, NIAH |
| Repeats / measurements | 3 repeats per arm/dataset; 24 measurements |
| Prompt-token mean | 15,491-23,231 |
| Evidence file | [`v1_benchmark.json`](v1_benchmark.json) |
| Primary-table mismatch | 3 repeats, 100-token completions, and prompt-token means 15,491-23,231 rather than the primary parallel-8/256-token/8k-16k-32k/disk-cache protocol |

## Main Latency Results

| Dataset | Baseline p50 TTFT | Cachet p50 TTFT | TTFT speedup | Baseline p50 TTC | Cachet p50 TTC | TTC speedup | Repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| biography | 4.460 s | 0.950 s | 4.693x | 6.901 s | 3.388 s | 2.037x | 3 |
| hotpotqa | 8.116 s | 1.343 s | 6.043x | 10.816 s | 4.047 s | 2.673x | 3 |
| musique | 7.609 s | 1.632 s | 4.662x | 10.287 s | 4.310 s | 2.387x | 3 |
| niah | 8.114 s | 1.372 s | 5.915x | 10.816 s | 4.071 s | 2.657x | 3 |

## Quality Results

| Dataset | Baseline `answer_found_rate` | Cachet `answer_found_rate` | `answer_found_rate` delta | Baseline `exact_match_rate` | Cachet `exact_match_rate` | `exact_match_rate` delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| biography | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hotpotqa | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| musique | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| niah | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Memory / Footprint

| Metric | Value |
| --- | --- |
| Peak GPU memory | Not measured |
| CPU RSS | Not measured |
| Cache-resident footprint | Not measured |
| Storage throughput | Not measured in this serving benchmark |

## Limitations

| Limitation | Current state |
| --- | --- |
| Target status | Compatibility result; g6/L4 is the primary target |
| Model coverage | Qwen3 4B Instruct only |
| Method coverage | Vanilla external KV only |
| Memory | Serving peak GPU memory, CPU RSS, and KV cache footprint are not measured |

## Provenance

Sanitized evidence is committed beside this README:

- [`v1_benchmark.json`](v1_benchmark.json)
- [`databricks_run_status.json`](databricks_run_status.json)

The matching Databricks audit mirror is
[`../../../databricks/vllm-qwen3-4b-g5-a10g-vanilla-kv/`](../../../databricks/vllm-qwen3-4b-g5-a10g-vanilla-kv/).
