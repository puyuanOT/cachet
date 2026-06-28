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
| Quality metric in this table | Prepared-suite answer-found containment plus strict exact match |
| Current evidence | [`appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/) |

The prepared-suite quality rows are smoke checks, not full benchmark accuracy.
Each dataset/context pair currently uses one prepared example repeated eight
times. `Answer-found` only checks whether the expected short answer appears
somewhere in a verbose forced-256 output; strict exact match is also shown
because it catches the over-generation visible in these runs.

## Main Latency And Resource Table

Latency values are seconds. Percentiles are computed over 32 successful
request-level measurements per method/context pair when a row is complete:
four prepared inputs times eight repeats.

`P50 decode tok/s` is computed per request as
`completion_tokens / (TTC - TTFT)`, then summarized across request-level
measurements. `vLLM KV capacity` is derived from the logged 237,728 GPU
KV-cache tokens divided by the nominal context length; vLLM directly reports
7.25x maximum concurrency for 32,768-token requests. The benchmark load is
still capped at 8 in-flight requests by `--max-num-seqs=8`.

`Accounted GPU memory` is the vLLM-log total of 2.71 GiB model-load memory,
0.26 GiB CUDA graph-capture memory, and 16.32 GiB available KV-cache memory,
or 19.29 GiB. This is the best committed evidence for overall GPU memory use;
the runs did not sample a true process-level `nvidia-smi` peak. The configured
vLLM GPU-memory budget is about 20 GiB from `gpu_memory_utilization=0.9` on the
A10G.

| Method | Input context | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | P50 decode tok/s | vLLM KV capacity | Accounted GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | 8k | 2.166 | 2.737 | 14.507 | 15.080 | 20.742 | 29.02x | 19.29 GiB |
| Baseline, no precomputed KV | 16k | 6.047 | 6.644 | 20.728 | 21.324 | 17.437 | 14.51x | 19.29 GiB |
| Baseline, no precomputed KV | 32k | 16.622 | 16.971 | 35.361 | 35.714 | 13.659 | 7.25x | 19.29 GiB |
| Cachet + vanilla KV | 8k | 0.389 | 0.422 | 12.730 | 12.752 | 20.732 | 29.02x | 19.29 GiB |
| Cachet + vanilla KV | 16k | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 14.51x | 19.29 GiB |
| Cachet + vanilla KV | 32k | 0.677 | 1.122 | 19.705 | 20.000 | 13.487 | 7.25x | 19.29 GiB |
| Cachet + KV Packet | 8k |  |  |  |  |  |  |  |
| Cachet + KV Packet | 16k |  |  |  |  |  |  |  |
| Cachet + KV Packet | 32k |  |  |  |  |  |  |  |

`Cachet + KV Packet` is not implemented yet. Its rows are kept so the main
table shape is stable as new methods arrive.

The server logs still include observed scheduler state and KV-pool use in the
appendix evidence, but the main table reports capacity rather than one sampled
scheduler snapshot.

## Prepared Dataset Quality Table

Each cell is `answer-found / strict EM`. The all-1.00 answer-found values should
not be read as real dataset scores: the current suite is intentionally tiny,
answer-found is a containment check, and strict EM is 0.00 for every completed
main row because the forced-256 outputs are verbose.

| Method | Input context | Prepared examples per dataset | Repeats per example | Biography | HotpotQA | MusiQue | NIAH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | 8k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Baseline, no precomputed KV | 16k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Baseline, no precomputed KV | 32k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Cachet + vanilla KV | 8k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Cachet + vanilla KV | 16k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Cachet + vanilla KV | 32k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Cachet + KV Packet | 8k |  |  |  |  |  |  |
| Cachet + KV Packet | 16k |  |  |  |  |  |  |
| Cachet + KV Packet | 32k |  |  |  |  |  |  |

## Document KV Precision Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, vLLM `0.23.0`,
`g5.8xlarge`, 16k input context, 8 requests in flight, forced 256-token
decode, local disk handoff bundles, static-salt prewarm, and vLLM runtime KV
dtype `fp8_e5m2`.

This ablation varies the document KV payload stored on disk. GPU KV residency
is still governed by the vLLM runtime KV dtype unless the serving engine gains
native packed-Q4 KV pages.

| Document KV payload | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | P50 decode tok/s | Answer-found / strict EM | Cache footprint | vLLM KV capacity | Accounted GPU memory | Peak GPU process memory | CPU RSS / host RAM | Notes |
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

| Storage tier | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | P50 decode tok/s | Cache footprint | vLLM KV capacity | Accounted GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RAM |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |
| Disk | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 4.92 GB | 14.51x | 19.29 GiB |  | Current default |
| Unity Catalog |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |
| Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Hardware Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, vLLM
`0.23.0`, 16k input context, disk cache, 8 requests in flight, forced
256-token decode.

| Hardware | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | P50 decode tok/s | Cache footprint | vLLM KV capacity | Accounted GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AWS g5/A10G, `g5.8xlarge` | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 4.92 GB | 14.51x | 19.29 GiB |  | Current default |
| AWS g6/L4, `g6.8xlarge` |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Serving Platform Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV,
`g5.8xlarge`, 16k input context, disk cache, 8 requests in flight, forced
256-token decode.

| Serving platform | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | P50 decode tok/s | Cache footprint | vLLM KV capacity | Accounted GPU memory | CPU RSS / host RAM | Notes |
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
