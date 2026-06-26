# Current Benchmark Tables

This page is the public benchmark appendix for Cachet. It is organized like a
paper table: one main comparison table first, then ablations. It does not
invent missing numbers. A blank numeric cell means not measured yet; not zero.

The committed appendix evidence at
[`../appendix/existing-results/`](../appendix/existing-results/) does not match
the fixed main-table configuration below. Treat it as prior evidence, not as
the source for the empty main-table cells. Rows marked `not measured yet` are
not benchmarked yet under the stated configuration.

## Main Table Configuration

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct, `qwen3:4b-instruct` |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |
| Input context lengths | 8k, 16k, 32k tokens |
| Cache location for Cachet methods | Local disk, not Unity Catalog |
| Datasets / score columns | Biography, HotpotQA, MusiQue, NIAH |
| Baseline | No precomputed KV cache |
| Cachet methods | Vanilla external KV; KV Packet planned |

## Main Performance Table

| Method | Input context | Cache location | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score | Status / evidence |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline, no precomputed KV | 8k | n/a |  |  |  |  |  |  |  |  | Not measured yet under the fixed main-table configuration |
| Baseline, no precomputed KV | 16k | n/a |  |  |  |  |  |  |  |  | Not measured yet under the fixed main-table configuration |
| Baseline, no precomputed KV | 32k | n/a |  |  |  |  |  |  |  |  | Not measured yet under the fixed main-table configuration |
| Cachet + vanilla KV | 8k | Disk |  |  |  |  |  |  |  |  | Not measured yet under the fixed main-table configuration |
| Cachet + vanilla KV | 16k | Disk |  |  |  |  |  |  |  |  | Not measured yet under the fixed main-table configuration |
| Cachet + vanilla KV | 32k | Disk |  |  |  |  |  |  |  |  | Not measured yet under the fixed main-table configuration |
| Cachet + KV Packet | 8k | Disk |  |  |  |  |  |  |  |  | not implemented yet |
| Cachet + KV Packet | 16k | Disk |  |  |  |  |  |  |  |  | not implemented yet |
| Cachet + KV Packet | 32k | Disk |  |  |  |  |  |  |  |  | not implemented yet |

Score columns should use the benchmark's chosen task metric for each dataset.
Existing appendix evidence reports `answer_found_rate` and `exact_match_rate`,
but no committed run currently records those scores under the fixed
main-table configuration.

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

| Cache location | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score | Status / evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RAM |  |  |  |  |  |  |  |  | Not measured yet |
| Disk |  |  |  |  |  |  |  |  | Not measured yet |
| Unity Catalog |  |  |  |  |  |  |  |  | Not measured yet |
| Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  |  |  | Not measured yet |

## Hardware Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Serving engine | vLLM |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Cache location | Disk |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |

| Hardware | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score | Status / evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AWS g5/A10G, `g5.8xlarge` |  |  |  |  |  |  |  |  | Not measured yet under this ablation configuration |
| AWS g6/L4, `g6.8xlarge` |  |  |  |  |  |  |  |  | Not measured yet under this ablation configuration |

## Serving Platform Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Cache location | Disk |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |

| Serving platform | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score | Status / evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM |  |  |  |  |  |  |  |  | Not measured yet under this ablation configuration |
| SGLang |  |  |  |  |  |  |  |  | Not measured yet under this ablation configuration |

## Resource Utilization

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Default engine / hardware | vLLM on AWS g5/A10G, `g5.8xlarge` |
| Default context / output | 16k input context, 256 emitted tokens |
| Request parallelism | 8 requests in flight |

| Method or ablation row | Cache location | Peak GPU memory | GPU utilization | CPU RSS / RAM | Disk read throughput | Network / UC read throughput | Cache footprint | Status / evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline, no precomputed KV | n/a |  |  |  |  |  |  | Not measured yet |
| Cachet + vanilla KV | Disk |  |  |  |  |  |  | Not measured yet |
| Cachet + vanilla KV | RAM |  |  |  |  |  |  | Not measured yet |
| Cachet + vanilla KV | Unity Catalog |  |  |  |  |  |  | Not measured yet |
| Cachet + vanilla KV | Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  | Not measured yet |
| Cachet + KV Packet | Disk |  |  |  |  |  |  | not implemented yet |

## Appendix Evidence

These committed results remain useful, but they do not satisfy the main-table
configuration because they used different prompt-token ranges, repeat counts,
output lengths, cache/storage assumptions, hardware, or serving paths.

| Appendix result | What it shows | Why it is not in the main table |
| --- | --- | --- |
| [`vllm-qwen3-4b-g6-l4-vanilla-kv`](../appendix/existing-results/vllm-qwen3-4b-g6-l4-vanilla-kv/) | vLLM + Qwen3 4B + g6/L4 vanilla KV speedup evidence: 5.27x-6.97x TTFT speedup, answer-found delta `0.0` | g6/L4, 3 repeats, prompt-token means 15,491-23,231, 100-token completions, not the fixed g5/parallel-8/256-token/disk-cache table |
| [`vllm-qwen3-4b-g5-a10g-vanilla-kv`](../appendix/existing-results/vllm-qwen3-4b-g5-a10g-vanilla-kv/) | vLLM + Qwen3 4B + g5/A10G compatibility evidence: 4.66x-6.04x TTFT speedup, answer-found delta `0.0` | 3 repeats, prompt-token means 15,491-23,231, 100-token completions, not the fixed 8k/16k/32k parallel-8 table |
| [`sglang-qwen3-4b-g6-l4-vanilla-kv-prepared`](../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-prepared/) | SGLang HiCache correctness/cache-hit evidence; no speedup on short prepared prompts | SGLang on g6/L4, short prompts, 2 repeats, not the fixed vLLM/g5/disk-cache main table |
| [`sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah`](../appendix/existing-results/sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/) | Small SGLang synthetic NIAH cache-hit check | One tiny synthetic prompt, not the four-dataset main table |
| [`storage-g6-l4-reader-throughput`](../appendix/existing-results/storage-g6-l4-reader-throughput/) | Memory, disk, and Unity Catalog reader throughput over a 256 MiB workload | Storage-reader throughput only, not serving TTFT/TTC or memory utilization |
| [`native-engine-g6-l4-vllm-sglang-vanilla-kv`](../appendix/existing-results/native-engine-g6-l4-vllm-sglang-vanilla-kv/) | vLLM/SGLang native connector probes copied 48 tokens / 3,538,944 bytes | Integration and copied-byte evidence, not serving latency or quality |

Use [`../databricks/CURRENT.md`](../databricks/CURRENT.md) only when you need
QA run IDs or release-source mirrors.
