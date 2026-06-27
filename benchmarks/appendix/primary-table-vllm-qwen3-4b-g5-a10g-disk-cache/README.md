# Primary Table Evidence: vLLM Qwen3 4B on g5/A10G

This folder contains the sanitized evidence backing the primary benchmark table
in the [benchmark root](../../). It matches the fixed primary-table
configuration: Qwen3-4B-Instruct, vLLM, AWS g5/A10G `g5.8xlarge`, 8 requests in
flight, 256 emitted tokens, and Cachet handoffs stored on local disk.

## Table Configuration

| Field | Value |
| --- | --- |
| Model | Qwen3-4B-Instruct, `qwen3:4b-instruct` |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |
| Input context lengths | 8k, 16k, 32k tokens |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Score metric | `answer_found_rate` |
| Baseline | No precomputed KV cache |
| Cachet method | Cachet + vanilla KV |
| Cache residency | Local disk under `/local_disk0`, not Unity Catalog |
| Source commit | `64b602a` |
| DBFS staging root | `dbfs:/benchmarks/cachet/primary-table-64b602a-20260626_232950` |

## Main Result Table

Latency values are seconds. Percentiles are computed over 32 successful
request-level measurements for each method/context pair: four datasets times
eight repeats.

| Method | Input context | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | 8k | 4.22 | 9.93 | 20.53 | 28.15 | 1.00 | 1.00 | 1.00 | 1.00 |
| Baseline, no precomputed KV | 16k | 25.40 | 33.43 | 41.32 | 58.02 | 1.00 | 1.00 | 1.00 | 1.00 |
| Baseline, no precomputed KV | 32k | 97.98 | 98.61 | 117.05 | 117.07 | 1.00 | 1.00 | 1.00 | 1.00 |
| Cachet + vanilla KV | 8k | 7.24 | 8.01 | 17.42 | 18.24 | 1.00 | 1.00 | 1.00 | 1.00 |
| Cachet + vanilla KV | 16k | 21.44 | 32.42 | 28.68 | 43.40 | 1.00 | 1.00 | 1.00 | 1.00 |
| Cachet + vanilla KV | 32k | 58.10 | 62.45 | 71.69 | 72.01 | 1.00 | 1.00 | 1.00 | 1.00 |

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
| Primary-table comparability | Matches the primary-table configuration for baseline and Cachet + vanilla KV |
| Method coverage | KV Packet is not implemented yet |
| Context coverage | Covers 8k, 16k, and 32k input context lengths |
| Resource metrics | Peak GPU memory, GPU utilization, CPU RSS/RAM, disk throughput, network throughput, and cache footprint were not measured |
| Dataset scope | Uses one synthetic prepared example per dataset/context pair |

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
