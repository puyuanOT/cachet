# Cachet Benchmarks

This directory is the public benchmark appendix for Cachet. It presents the
current benchmark protocol only: Qwen3-4B-Instruct with 4-bit model weights,
Q8 document KV, shared GPU prefix references, and private KV for the user
question plus generated tokens. The main latency table measures cold document
KV hydration: the Cachet rows load persisted document KV from local disk into
GPU-resident serving-engine KV state inside the measured request path.

Historical benchmark folders were removed from this directory to avoid mixing
incompatible measurements with the current protocol. Older records remain
recoverable from git history and Databricks run provenance when needed for
audit work.

Blank numeric cells mean the row has not been measured under the current
protocol yet. A blank cell is not a zero.

## Shared Main Table Configuration

The configuration below applies to both the Main Latency And Resource Table and
the Benchmark Dataset Score Table unless a table caption explicitly says
otherwise. Input-context length is varied only for the latency/resource table;
dataset scores are evaluated over the selected dataset samples.

| Field | Fixed value |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` served as `qwen3:4b-instruct` |
| Model weights | vLLM `--quantization bitsandbytes` 4-bit weights |
| Serving engine | vLLM `0.23.0` |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for latency | Forced 256-token decode with `max_tokens=256` and `ignore_eos=true` |
| Latency input context lengths | 8k, 16k, and 32k prepared prompts |
| Default Cachet method | Vanilla external KV |
| Default document KV precision | Q8, represented as `fp8_e5m2` payloads |
| vLLM runtime KV dtype | `fp8_e5m2` |
| Cache residency | Local disk handoff bundles; Cachet rows hydrate document KV from disk during measured requests |
| Prefix-cache policy | vLLM prefix caching enabled with per-request `cache_salt` isolation for latency rows |
| Runtime KV ownership | Shared GPU KV for the loaded document/system prefix during each request; private KV for request-specific prompt suffix and generated tokens |
| Score datasets | Biography, HotpotQA, MusiQue, NIAH |
| Score metric | Full-dataset task score; blank until full-dataset runs complete |
| Warm-prefix canary evidence | [`appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/) |

The appendix currently includes prepared-suite warm-prefix smoke checks only.
Those checks are not cold-hydrate latency rows, are not full benchmark scores,
and are not copied into the main score table.

## Main Latency And Resource Table

Latency values are seconds.

| Method | Input context | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Max Serving Concurrency | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8k |  |  |  |  |  | 29.02x |  |
| Baseline | 16k |  |  |  |  |  | 14.51x |  |
| Baseline | 32k |  |  |  |  |  | 7.25x |  |
| vanilla&nbsp;KV | 8k |  |  |  |  |  | 29.02x |  |
| vanilla&nbsp;KV | 16k |  |  |  |  |  | 14.51x |  |
| vanilla&nbsp;KV | 32k |  |  |  |  |  | 7.25x |  |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | 8k |  |  |  |  |  |  |  |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | 16k |  |  |  |  |  |  |  |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | 32k |  |  |  |  |  |  |  |

Caption: `Baseline` means vLLM receives the complete prompt and computes KV for
the system prompt, documents, user question, and generated tokens at request
time. `vanilla KV` means Cachet reuses precomputed raw KV for the reusable
system/document prefix by reading the persisted handoff bundle from local disk,
hydrating those pages into vLLM-managed GPU KV state during the measured
request, and leaving only the request-specific prompt suffix plus generated
tokens as private runtime KV. `KV Packet` is a planned method and is not
implemented yet.

Latency rows are generated with `request_parallelism=8`: the benchmark runner
issues up to eight concurrent requests while collecting request-level TTFT and
TTC measurements. Cold-hydrate rows must use per-request
`cache_salt` isolation so repeated examples do not reuse vLLM prefix-cache
blocks across measured requests. Publication rows should use at least 512
successful request-level measurements per method/context pair at the same
concurrency.

`cache_salt` is the namespace vLLM includes in its prefix-cache key. A static
salt lets identical prefixes share already-resident KV blocks across requests;
that is useful for warm-prefix ablations but does not measure disk-to-GPU
hydrate cost. The main table uses per-request salt values, so vLLM prefix
caching remains enabled but cannot turn repeated measurements into warm prefix
hits.

`P50 tok/s` is computed per request as
`completion_tokens / (TTC - TTFT)`, then summarized across request-level
measurements. `Max Serving Concurrency` is derived from the logged 237,728 GPU
KV-cache tokens divided by the nominal context length; vLLM directly reports
7.25x maximum concurrency for 32,768-token requests. The benchmark load is
still capped at 8 in-flight requests by `--max-num-seqs=8`.

`Peak GPU memory` is populated only from sampled runtime telemetry such as
`nvidia-smi` peak process/device memory during the benchmark run. The
warm-prefix canary runs reported only server-level vLLM component accounting
and did not sample a true process-level peak, so this column remains blank
until the cold-hydrate 512-measurement reruns complete.

The server logs still include observed scheduler state and KV-pool use in the
appendix evidence, but the main table reports configured capacity rather than
one sampled scheduler snapshot.

## Benchmark Dataset Score Table

| Method | Biography score | HotpotQA score | MusiQue score | NIAH score |
| --- | ---: | ---: | ---: | ---: |
| Baseline |  |  |  |  |
| vanilla&nbsp;KV |  |  |  |  |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) |  |  |  |  |

Caption: scores are reserved for full-dataset evaluations over all selected
samples for Biography, HotpotQA, MusiQue, and NIAH. The score table has no
input-context column because input-context length is a latency stress dimension,
not a separate full-dataset scoring condition in the main table. The previous
all-1.00 answer-found values were one-example smoke checks and are retained
only in the appendix evidence; they are not official dataset scores.

## Document KV Precision Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, vLLM `0.23.0`,
`g5.8xlarge`, 16k input context, 8 requests in flight, forced 256-token
decode, local disk handoff bundles, cold disk-to-GPU hydrate, per-request
`cache_salt` isolation, and vLLM runtime KV dtype `fp8_e5m2`.

This ablation varies the document KV payload stored on disk. GPU KV residency
is still governed by the vLLM runtime KV dtype unless the serving engine gains
native packed-Q4 KV pages.

| Document KV payload | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Answer-found / strict EM | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bf16 |  |  |  |  |  |  | 9.83 GB | 14.51x |  |  | Warm-prefix canary showed a quality failure under FP8 runtime KV; rerun required under cold-hydrate protocol |
| Q8 (`fp8_e5m2`) |  |  |  |  |  |  | 4.92 GB | 14.51x |  |  | Default document KV precision; cold-hydrate latency rerun pending |
| Q4 packed document KV |  |  |  |  |  |  |  |  |  |  | Implementation pending; requires packed-Q4 payload layout and provider dequant or native serving-engine Q4 KV support |

Peak GPU process memory, GPU utilization, CPU RSS, and host RAM were not
sampled in these runs. vLLM server logs report 2.71 GiB model-load memory,
0.26 GiB CUDA graph-capture memory, 16.32 GiB available KV-cache memory,
237,728 GPU KV-cache tokens, 7.25x maximum concurrency for 32,768-token
requests, and about 20 GiB configured GPU-memory budget after loading the
4-bit Qwen3-4B model.

## Storage Tier Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, vLLM
`0.23.0`, `g5.8xlarge`, 16k input context, 8 requests in flight, forced
256-token decode, and cold disk-to-GPU hydrate unless the storage tier itself
is RAM.

| Storage tier | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RAM |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |
| Disk |  |  |  |  |  | 4.92 GB | 14.51x |  |  | Current default; cold-hydrate latency rerun pending |
| Unity Catalog |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |
| Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Hardware Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, vLLM
`0.23.0`, 16k input context, disk cache, 8 requests in flight, forced
256-token decode.

| Hardware | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AWS g5/A10G, `g5.8xlarge` |  |  |  |  |  | 4.92 GB | 14.51x |  |  | Current default; cold-hydrate latency rerun pending |
| AWS g6/L4, `g6.8xlarge` |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Serving Platform Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV,
`g5.8xlarge`, 16k input context, disk cache, 8 requests in flight, forced
256-token decode.

| Serving platform | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM |  |  |  |  |  | 4.92 GB | 14.51x |  |  | Current default; cold-hydrate latency rerun pending |
| SGLang |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Directory Layout

| Folder | Purpose |
| --- | --- |
| [`appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/) | Warm-prefix canary evidence and Databricks provenance for the current Q4/Q8 configuration |
| [`databricks/`](databricks/) | Notes for Databricks provenance; historical committed mirrors have been removed |
| [`_template/`](_template/) | Required table shape for future public benchmark result folders |

Do not add raw Databricks Jobs API responses, credentials, package wheels,
driver logs, generated datasets, prompt payload blobs, or local scratch output
to this directory.
