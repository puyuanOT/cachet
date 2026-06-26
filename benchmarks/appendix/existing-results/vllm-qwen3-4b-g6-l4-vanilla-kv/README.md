# vLLM Qwen3 4B On g6/L4 With Vanilla KV

Appendix evidence for a prior Cachet vLLM speedup benchmark on g6/L4 hardware.
It is not a row in the fixed main table in [benchmark root](../../../).

## Experimental Setup

| Field | Value |
| --- | --- |
| Engine | vLLM |
| Model | `qwen3:4b-instruct` |
| Hardware | AWS g6/L4, `g6.8xlarge` |
| Method | Vanilla external KV |
| Baseline arm | `baseline_prefill` |
| Cache arm | `document_kv_cache` |
| Dataset scope | Biography, HotpotQA, MusiQue, NIAH |
| Repeats / measurements | 3 repeats per arm/dataset; 24 measurements |
| Prompt-token mean | 15,491-23,231 |
| Evidence file | [`v1_benchmark.json`](v1_benchmark.json) |
| Main-table mismatch | g6/L4, 3 repeats, 100-token completions, and prompt-token means 15,491-23,231 rather than fixed g5/parallel-8/256-token/8k-16k-32k/disk-cache |

## Main Latency Results

| Dataset | Baseline p50 TTFT | Cachet p50 TTFT | TTFT speedup | Baseline p50 TTC | Cachet p50 TTC | TTC speedup | Repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| biography | 4.841 s | 0.919 s | 5.267x | 9.205 s | 5.285 s | 1.742x | 3 |
| hotpotqa | 8.878 s | 1.273 s | 6.972x | 13.807 s | 6.202 s | 2.226x | 3 |
| musique | 8.433 s | 1.583 s | 5.326x | 13.300 s | 6.451 s | 2.062x | 3 |
| niah | 9.188 s | 1.332 s | 6.896x | 14.126 s | 6.267 s | 2.254x | 3 |

## Quality Results

| Dataset | Baseline answer-found | Cachet answer-found | Answer-found delta | Baseline exact-match | Cachet exact-match | Exact-match delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| biography | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hotpotqa | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| musique | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| niah | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |

`answer_found_rate` is the current quality gate. Exact match is a separate raw
field and is `0.0` for both arms in this evidence.

## Memory / Footprint

| Metric | Value |
| --- | --- |
| Peak GPU memory | not measured |
| CPU RSS | not measured |
| Cache-resident footprint | not measured |
| Storage throughput | not measured in this serving benchmark |

## Limitations

| Limitation | Current state |
| --- | --- |
| Model coverage | Qwen3 4B Instruct only |
| Method coverage | Vanilla external KV only |
| Repeat count | 3 repeats per arm/dataset |
| Memory | Serving peak GPU memory, CPU RSS, and cache footprint are not measured |

## Provenance

Sanitized evidence is committed beside this README:

- [`v1_benchmark.json`](v1_benchmark.json)
- [`databricks_run_status.json`](databricks_run_status.json)
- [`release_evidence.json`](release_evidence.json)

The matching Databricks audit mirror is
[`../../../databricks/vllm-qwen3-4b-g6-l4-vanilla-kv/`](../../../databricks/vllm-qwen3-4b-g6-l4-vanilla-kv/).
