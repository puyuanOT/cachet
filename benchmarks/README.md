# Cachet Benchmarks

This directory is the public benchmark appendix for Cachet. It presents the
current benchmark protocol only: Qwen3-4B-Instruct with 4-bit model weights,
Q8 document KV, shared GPU prefix references, and private KV for the user
question plus generated tokens.

Historical benchmark folders were removed from this directory to avoid mixing
incompatible measurements with the current protocol. Older records remain
recoverable from git history and Databricks run provenance when needed for
audit work.

Blank numeric cells mean the row has not been measured under the current
protocol yet. A blank cell is not a zero.

## Main Table Configuration

| Field | Fixed value |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` served as `qwen3:4b-instruct` |
| Model weights | vLLM `--quantization bitsandbytes` 4-bit weights |
| Serving engine | vLLM `0.23.0` |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for latency | Forced 256-token decode with `max_tokens=256` and `ignore_eos=true` |
| Input context lengths | 8k, 16k, and 32k prepared prompts |
| Default Cachet method | Vanilla external KV |
| Default document KV precision | Q8, represented as `fp8_e5m2` payloads |
| vLLM runtime KV dtype | `fp8_e5m2` |
| Cache residency | Local disk handoff bundles, prewarmed into the vLLM prefix cache |
| Prefix sharing | Static `cache_salt`, vLLM prefix caching enabled |
| Runtime KV ownership | Shared GPU KV for the cached document/system prefix; private KV for request-specific prompt suffix and generated tokens |
| Datasets / quality table | Biography, HotpotQA, MusiQue, NIAH |
| Quality metric in this table | Full-dataset task score; blank until full-dataset runs complete |
| Current evidence | [`appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/) |

The appendix currently includes prepared-suite smoke checks only. Those checks
are not full benchmark scores and are not copied into the main score table.

## Main Latency And Resource Table

Latency values are seconds.

| Method | Input context | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | P50 decode tok/s | vLLM max concurrency | Accounted GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8k | 2.166 | 2.737 | 14.507 | 15.080 | 20.742 | 29.02x | 19.29 GiB |
| Baseline | 16k | 6.047 | 6.644 | 20.728 | 21.324 | 17.437 | 14.51x | 19.29 GiB |
| Baseline | 32k | 16.622 | 16.971 | 35.361 | 35.714 | 13.659 | 7.25x | 19.29 GiB |
| vanilla KV | 8k | 0.389 | 0.422 | 12.730 | 12.752 | 20.732 | 29.02x | 19.29 GiB |
| vanilla KV | 16k | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 14.51x | 19.29 GiB |
| vanilla KV | 32k | 0.677 | 1.122 | 19.705 | 20.000 | 13.487 | 7.25x | 19.29 GiB |
| [KV Packet](https://arxiv.org/abs/2604.13226) | 8k |  |  |  |  |  |  |  |
| [KV Packet](https://arxiv.org/abs/2604.13226) | 16k |  |  |  |  |  |  |  |
| [KV Packet](https://arxiv.org/abs/2604.13226) | 32k |  |  |  |  |  |  |  |

Caption: `Baseline` means vLLM receives the complete prompt and computes KV for
the system prompt, documents, user question, and generated tokens at request
time. `vanilla KV` means Cachet reuses precomputed raw KV for the reusable
system/document prefix, prewarms those pages into vLLM shared GPU prefix
references, and leaves only the request-specific prompt suffix plus generated
tokens as private runtime KV. `KV Packet` is a planned method and is not
implemented yet.

The latency rows were generated with `request_parallelism=8`: the benchmark
runner issued up to eight concurrent requests while collecting request-level
TTFT and TTC measurements. Each complete row currently has 32 successful
request-level measurements, from four prepared inputs repeated eight times.
That is enough for a canary comparison but not enough for a publication-grade
P95; publication rows should use at least 1,000 successful request-level
measurements per method/context pair at the same concurrency.

`P50 decode tok/s` is computed per request as
`completion_tokens / (TTC - TTFT)`, then summarized across request-level
measurements. `vLLM max concurrency` is derived from the logged 237,728 GPU
KV-cache tokens divided by the nominal context length; vLLM directly reports
7.25x maximum concurrency for 32,768-token requests. The benchmark load is
still capped at 8 in-flight requests by `--max-num-seqs=8`.

`Accounted GPU memory` is server-level vLLM component telemetry from the shared
process, so it is identical for rows produced by the same loaded server. It is
the sum of 2.71 GiB model-load memory, 0.26 GiB CUDA graph-capture memory, and
16.32 GiB available KV-cache memory, or 19.29 GiB. The runs did not sample a
true process-level `nvidia-smi` peak. The configured vLLM GPU-memory budget is
about 20 GiB from `gpu_memory_utilization=0.9` on the A10G.

The server logs still include observed scheduler state and KV-pool use in the
appendix evidence, but the main table reports configured capacity rather than
one sampled scheduler snapshot.

## Benchmark Dataset Score Table

| Method | Input context | Biography score | HotpotQA score | MusiQue score | NIAH score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8k |  |  |  |  |
| Baseline | 16k |  |  |  |  |
| Baseline | 32k |  |  |  |  |
| vanilla KV | 8k |  |  |  |  |
| vanilla KV | 16k |  |  |  |  |
| vanilla KV | 32k |  |  |  |  |
| [KV Packet](https://arxiv.org/abs/2604.13226) | 8k |  |  |  |  |
| [KV Packet](https://arxiv.org/abs/2604.13226) | 16k |  |  |  |  |
| [KV Packet](https://arxiv.org/abs/2604.13226) | 32k |  |  |  |  |

Caption: scores are reserved for full-dataset evaluations over all selected
samples for Biography, HotpotQA, MusiQue, and NIAH. The previous all-1.00
answer-found values were one-example smoke checks and are retained only in the
appendix evidence; they are not official dataset scores.

## Document KV Precision Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, vLLM `0.23.0`,
`g5.8xlarge`, 16k input context, 8 requests in flight, forced 256-token
decode, local disk handoff bundles, static-salt prewarm, and vLLM runtime KV
dtype `fp8_e5m2`.

This ablation varies the document KV payload stored on disk. GPU KV residency
is still governed by the vLLM runtime KV dtype unless the serving engine gains
native packed-Q4 KV pages.

| Document KV payload | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | P50 decode tok/s | Answer-found / strict EM | Cache footprint | vLLM max concurrency | Accounted GPU memory | Peak GPU process memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bf16 | 0.539 | 0.646 | 15.224 | 15.336 | 17.425 | 0.00 / 0.00 | 9.83 GB | 14.51x | 19.29 GiB |  |  | Quality failure under FP8 runtime KV: measured outputs did not contain expected answers |
| Q8 (`fp8_e5m2`) | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 1.00 / 0.00 | 4.92 GB | 14.51x | 19.29 GiB |  |  | Default document KV precision |
| Q4 packed document KV |  |  |  |  |  |  |  |  |  |  |  | Implementation pending; requires packed-Q4 payload layout and provider dequant or native serving-engine Q4 KV support |

Peak GPU process memory, GPU utilization, CPU RSS, and host RAM were not
sampled in these runs. vLLM server logs report 2.71 GiB model-load memory,
0.26 GiB CUDA graph-capture memory, 16.32 GiB available KV-cache memory,
237,728 GPU KV-cache tokens, 7.25x maximum concurrency for 32,768-token
requests, and about 20 GiB configured GPU-memory budget after loading the
4-bit Qwen3-4B model.

## Storage Tier Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, vLLM
`0.23.0`, `g5.8xlarge`, 16k input context, 8 requests in flight, forced
256-token decode.

| Storage tier | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | P50 decode tok/s | Cache footprint | vLLM max concurrency | Accounted GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RAM |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |
| Disk | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 4.92 GB | 14.51x | 19.29 GiB |  | Current default |
| Unity Catalog |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |
| Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Hardware Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, vLLM
`0.23.0`, 16k input context, disk cache, 8 requests in flight, forced
256-token decode.

| Hardware | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | P50 decode tok/s | Cache footprint | vLLM max concurrency | Accounted GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AWS g5/A10G, `g5.8xlarge` | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 4.92 GB | 14.51x | 19.29 GiB |  | Current default |
| AWS g6/L4, `g6.8xlarge` |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Serving Platform Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV,
`g5.8xlarge`, 16k input context, disk cache, 8 requests in flight, forced
256-token decode.

| Serving platform | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | P50 decode tok/s | Cache footprint | vLLM max concurrency | Accounted GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 4.92 GB | 14.51x | 19.29 GiB |  | Current default |
| SGLang |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Directory Layout

| Folder | Purpose |
| --- | --- |
| [`appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/) | Current benchmark evidence and Databricks provenance for the default protocol |
| [`databricks/`](databricks/) | Notes for Databricks provenance; historical committed mirrors have been removed |
| [`_template/`](_template/) | Required table shape for future public benchmark result folders |

Do not add raw Databricks Jobs API responses, credentials, package wheels,
driver logs, generated datasets, prompt payload blobs, or local scratch output
to this directory.
