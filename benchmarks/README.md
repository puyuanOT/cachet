# Cachet Benchmarks

This directory is the public benchmark appendix for Cachet. It follows a
research-paper structure: one primary comparison table first, followed by
focused ablation tables. No values are inferred or estimated. A blank numeric
cell means not measured yet; not zero.

The committed appendix evidence at
[`appendix/existing-results/`](appendix/existing-results/) does not match the
primary-table configuration below. Treat it as prior evidence, not as the
source for the empty primary-table cells. Rows marked `pending measurement`
have not been measured under the stated configuration.

## Main Table Configuration

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct, `qwen3:4b-instruct` |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |
| Input context lengths | 8k, 16k, 32k tokens |
| KV cache residency for Cachet methods | Local disk, not Unity Catalog |
| Datasets / score columns | Biography, HotpotQA, MusiQue, NIAH |
| Baseline | No precomputed KV cache |
| Cachet methods | Vanilla external KV; KV Packet planned but not implemented yet |

## Main Performance Table

| Method | Input context | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score | Status / evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline, no precomputed KV | 8k |  |  |  |  |  |  |  |  | Pending measurement under the primary-table configuration |
| Baseline, no precomputed KV | 16k |  |  |  |  |  |  |  |  | Pending measurement under the primary-table configuration |
| Baseline, no precomputed KV | 32k |  |  |  |  |  |  |  |  | Pending measurement under the primary-table configuration |
| Cachet + vanilla KV | 8k |  |  |  |  |  |  |  |  | Pending measurement under the primary-table configuration |
| Cachet + vanilla KV | 16k |  |  |  |  |  |  |  |  | Pending measurement under the primary-table configuration |
| Cachet + vanilla KV | 32k |  |  |  |  |  |  |  |  | Pending measurement under the primary-table configuration |
| Cachet + KV Packet | 8k |  |  |  |  |  |  |  |  | Planned method; not implemented yet |
| Cachet + KV Packet | 16k |  |  |  |  |  |  |  |  | Planned method; not implemented yet |
| Cachet + KV Packet | 32k |  |  |  |  |  |  |  |  | Planned method; not implemented yet |

Dataset score columns should report the task metric selected by the benchmark
plan for each dataset. Existing appendix evidence reports `answer_found_rate`
and `exact_match_rate`, but no committed run currently records those scores
under the primary-table configuration.

## Storage Tier Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |

| Storage tier | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score | Status / evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RAM |  |  |  |  |  |  |  |  | Pending measurement |
| Disk |  |  |  |  |  |  |  |  | Pending measurement |
| Unity Catalog |  |  |  |  |  |  |  |  | Pending measurement |
| Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  |  |  | Pending measurement |

## Hardware Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Serving engine | vLLM |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Storage tier | Disk |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |

| Hardware | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score | Status / evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AWS g5/A10G, `g5.8xlarge` |  |  |  |  |  |  |  |  | Pending measurement under this ablation configuration |
| AWS g6/L4, `g6.8xlarge` |  |  |  |  |  |  |  |  | Pending measurement under this ablation configuration |

## Serving Platform Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Storage tier | Disk |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |

| Serving platform | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score | Status / evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM |  |  |  |  |  |  |  |  | Pending measurement under this ablation configuration |
| SGLang |  |  |  |  |  |  |  |  | Pending measurement under this ablation configuration |

## Resource Utilization

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Default engine / hardware | vLLM on AWS g5/A10G, `g5.8xlarge` |
| Default context / output | 16k input context, 256 emitted tokens |
| Request parallelism | 8 requests in flight |

| Experiment row | Storage tier | Peak GPU memory | GPU utilization | Peak CPU RSS / host RAM | Disk read throughput | Network / Unity Catalog read throughput | KV cache footprint | Status / evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline, no precomputed KV | N/A |  |  |  |  |  |  | Pending measurement |
| Cachet + vanilla KV | Disk |  |  |  |  |  |  | Pending measurement |
| Cachet + vanilla KV | RAM |  |  |  |  |  |  | Pending measurement |
| Cachet + vanilla KV | Unity Catalog |  |  |  |  |  |  | Pending measurement |
| Cachet + vanilla KV | Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  | Pending measurement |
| Cachet + KV Packet | Disk |  |  |  |  |  |  | Planned method; not implemented yet |

## Appendix Evidence

These committed results remain useful, but they do not satisfy the primary-table
configuration because they used different prompt-token ranges, repeat counts,
output lengths, cache/storage assumptions, hardware, or serving paths.

| Appendix result | Evidence summary | Configuration mismatch |
| --- | --- | --- |
| [`vllm-qwen3-4b-g6-l4-vanilla-kv`](appendix/existing-results/vllm-qwen3-4b-g6-l4-vanilla-kv/) | vLLM + Qwen3 4B + g6/L4 vanilla KV latency evidence: 5.27x-6.97x TTFT speedup, `answer_found_rate` delta `0.0` | g6/L4, 3 repeats, prompt-token means 15,491-23,231, 100-token completions, not the fixed g5/parallel-8/256-token/disk-cache table |
| [`vllm-qwen3-4b-g5-a10g-vanilla-kv`](appendix/existing-results/vllm-qwen3-4b-g5-a10g-vanilla-kv/) | vLLM + Qwen3 4B + g5/A10G compatibility evidence: 4.66x-6.04x TTFT speedup, `answer_found_rate` delta `0.0` | 3 repeats, prompt-token means 15,491-23,231, 100-token completions, not the fixed 8k/16k/32k parallel-8 table |
| [`sglang-qwen3-4b-g6-l4-vanilla-kv-prepared`](appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-prepared/) | SGLang HiCache correctness and cache-hit evidence; no latency improvement observed on short prepared prompts | SGLang on g6/L4, short prompts, 2 repeats, not the fixed vLLM/g5/disk-cache primary table |
| [`sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah`](appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/) | Minimal SGLang synthetic NIAH cache-hit check | Single minimal synthetic prompt, not the four-dataset primary table |
| [`storage-g6-l4-reader-throughput`](appendix/existing-results/storage-g6-l4-reader-throughput/) | Memory, disk, and Unity Catalog reader throughput over a 256 MiB workload | Storage-reader throughput only, not serving TTFT/TTC or memory utilization |
| [`native-engine-g6-l4-vllm-sglang-vanilla-kv`](appendix/existing-results/native-engine-g6-l4-vllm-sglang-vanilla-kv/) | vLLM/SGLang native connector probes copied 48 tokens / 3,538,944 bytes | Integration and copied-byte evidence, not serving latency or quality |

Use [`databricks/CURRENT.md`](databricks/CURRENT.md) only when you need QA run
IDs or release-source mirrors.

## Directory Layout

| Folder | Purpose |
| --- | --- |
| [`appendix/existing-results/`](appendix/existing-results/) | Committed evidence from configurations that do not match the primary-table protocol |
| [`databricks/`](databricks/) | Sanitized Databricks audit mirrors kept at stable paths for release evidence |
| [`_template/`](_template/) | Required table shape for future public benchmark result folders |
| [`vllm/`](vllm/) | Short index page for vLLM appendix evidence |
| [`sglang/`](sglang/) | Short index page for SGLang appendix evidence |
| [`storage/`](storage/) | Short index page for storage-reader appendix evidence |
| [`native-engine/`](native-engine/) | Short index page for native connector appendix evidence |

The [`databricks/`](databricks/) folder remains unmoved because
release-evidence JSON records refer to those paths. Do not put raw Databricks
Jobs API responses, credentials, package wheels, driver logs, generated
datasets, prompt payload blobs, or local scratch output in this directory.

Historical failed SGLang smoke attempts and superseded readiness artifacts live
under [`../docs/release-ops/benchmark-archive/`](../docs/release-ops/benchmark-archive/).
