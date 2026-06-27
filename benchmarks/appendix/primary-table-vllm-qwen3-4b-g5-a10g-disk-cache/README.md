# Baseline and Full-Logical-Prompt vLLM Evidence: Qwen3 4B on g5/A10G

This folder contains sanitized evidence referenced by the benchmark root. The
baseline rows match the fixed primary-table configuration: Qwen3-4B-Instruct,
vLLM, AWS g5/A10G `g5.8xlarge`, 8 requests in flight, and 256 emitted tokens.
The Cachet rows in this folder are full-logical-prompt vLLM handoff-path
measurements: they are useful appendix evidence, but they do not fill the
primary Cachet latency cells because the engine still received the full logical
prompt.

## Table Configuration

| Field | Value |
| --- | --- |
| Model | Qwen3-4B-Instruct, `qwen3:4b-instruct` |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Up to 256 generated tokens; requests were not forced to ignore EOS |
| Input context lengths | 8k, 16k, 32k tokens |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Score metric | Synthetic prepared-input `answer_found_rate`; full-dataset score not measured |
| Baseline | No precomputed KV cache |
| Cachet method | Cachet + vanilla KV |
| Cache residency | Local disk under `/local_disk0`, not Unity Catalog |
| Cachet prompt mode | Full logical prompt, not suffix-only runtime prompt |
| Source commit | `64b602a` |
| DBFS staging root | `dbfs:/benchmarks/cachet/primary-table-64b602a-20260626_232950` |

## Full-Logical-Prompt Result Table

Latency values are seconds. Percentiles are computed over 32 successful
request-level measurements for each method/context pair: four synthetic
prepared inputs times eight repeats. These values measure the current
end-to-end vLLM handoff path. They are not an isolated prefill-elision
microbenchmark: the Cachet requests still send the full logical prompt, and
the raw records report vLLM prompt usage equal to the 8k/16k/32k context size.
Only the baseline rows are used as primary-table latency values.

| Method | Input context | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography answer-found | HotpotQA answer-found | MusiQue answer-found | NIAH answer-found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | 8k | 4.22 | 9.93 | 20.53 | 28.15 | 1.00 | 1.00 | 1.00 | 1.00 |
| Baseline, no precomputed KV | 16k | 25.40 | 33.43 | 41.32 | 58.02 | 1.00 | 1.00 | 1.00 | 1.00 |
| Baseline, no precomputed KV | 32k | 97.98 | 98.61 | 117.05 | 117.07 | 1.00 | 1.00 | 1.00 | 1.00 |
| Cachet + vanilla KV | 8k | 7.24 | 8.01 | 17.42 | 18.24 | 1.00 | 1.00 | 1.00 | 1.00 |
| Cachet + vanilla KV | 16k | 21.44 | 32.42 | 28.68 | 43.40 | 1.00 | 1.00 | 1.00 | 1.00 |
| Cachet + vanilla KV | 32k | 58.10 | 62.45 | 71.69 | 72.01 | 1.00 | 1.00 | 1.00 | 1.00 |

The answer-found columns are synthetic sanity checks, not full-dataset
accuracy. The raw records also report low `exact_match_rate`, which is expected
because the model often emits the answer plus additional text.

## Resource Utilization

| Experiment row | Storage tier | Peak GPU memory | GPU utilization | Peak CPU RSS / host RAM | Disk read throughput | Network / Unity Catalog read throughput | KV cache footprint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | N/A |  |  |  |  |  |  |
| Cachet + vanilla KV | Disk |  |  |  |  |  |  |

Resource utilization was not collected in this run. The vLLM server logs and
Databricks driver logs were intentionally not committed.

## Limitations

| Limitation | Current state |
| --- | --- |
| Primary-table comparability | Baseline rows match the primary-table latency configuration; Cachet rows do not satisfy the suffix-only Cachet latency criterion |
| Method coverage | KV Packet is not implemented yet |
| Context coverage | Covers 8k, 16k, and 32k input context lengths |
| Resource metrics | Peak GPU memory, GPU utilization, CPU RSS/RAM, disk throughput, network throughput, and cache footprint were not measured |
| Dataset scope | Uses one synthetic prepared example per dataset/context pair; full-dataset scores are not measured |
| TTFT interpretation | Measures current end-to-end vLLM handoff-path latency, not a pure KV-load or prefill-skip microbenchmark |

## Provenance

| File | Contents |
| --- | --- |
| [`summary.json`](summary.json) | Aggregated table rows, configuration, evidence filenames, and Databricks run IDs |
| [`v1_benchmark_8k_baseline.json`](v1_benchmark_8k_baseline.json) | Raw V1 benchmark record for the 8k baseline row |
| [`v1_benchmark_16k_baseline.json`](v1_benchmark_16k_baseline.json) | Raw V1 benchmark record for the 16k baseline row |
| [`v1_benchmark_32k_baseline.json`](v1_benchmark_32k_baseline.json) | Raw V1 benchmark record for the 32k baseline row |
| [`v1_benchmark_8k_cachet_vanilla_kv.json`](v1_benchmark_8k_cachet_vanilla_kv.json) | Raw V1 benchmark record for the 8k Cachet + vanilla KV row |
| [`v1_benchmark_16k_cachet_vanilla_kv.json`](v1_benchmark_16k_cachet_vanilla_kv.json) | Raw V1 benchmark record for the 16k Cachet + vanilla KV row |
| [`v1_benchmark_32k_cachet_vanilla_kv.json`](v1_benchmark_32k_cachet_vanilla_kv.json) | Raw V1 benchmark record for the 32k Cachet + vanilla KV row |

Databricks run IDs: `563158142822504`, `353538553239170`,
`783822623402774`, `444278496547055`, `945864622976604`, and
`899697029840236`.
