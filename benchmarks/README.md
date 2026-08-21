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

Unsupported or unrun cells are marked `N/A` with a reason. They are not zeros.

> **Evidence status: descriptive / nonpublication-qualified.** The current
> Q4-weight/Q8-KV measurements are backed by a
> [compact sanitized record](appendix/main-vanilla-descriptive-evidence/), but
> the main method comparison does not pass the canonical canary gate. The
> isolated Baseline raw record structurally fails that generic gate because it
> has no cache arm; its hash-bound run-level telemetry is outside the per-arm
> resource schema. The
> matched Baseline-versus-Vanilla rows are therefore descriptive, not canary or
> publication claims. See the [evidence policy](../docs/evidence-policy.md).

Every current latency pair uses the same frozen model/tokenizer revision,
source, wheel, logical input bundle, request parallelism, decode settings, and
GPU-memory setting within that pair. Ablation tables preserve those pins except
for the factor explicitly named in their caption. The score diagnostic was run
from an earlier frozen source/wheel pair and records that identity separately;
it is never merged with the latency rows.

## Comparison And Measurement Rules

The main method table compares multiple methods under one fixed experiment
setting. Model and tokenizer revisions, engine, exact hardware shape, weight
and KV precision, storage and hydration boundary, logical samples, input/output
budgets, decoding options, concurrency, cache state, ordering, and seeds must
match across arms. By contrast, an ablation keeps one method and all other
settings fixed while varying exactly one declared factor, such as hardware,
quantization, storage tier, or serving platform. A run that changes several
factors is a separate setting comparison, not a one-variable ablation.

All arms start from the same logical documents, question, expected answer, and
decode budget. A method may transform its physical prompt, token sequence, or
artifact layout, but the transformation and version, physical token accounting,
and resulting artifact identity must be recorded. This permits legitimate
method-specific processing without hiding unequal inputs.

Online latency and resource tables use an explicitly declared serving boundary.
Offline training, artifact generation, checkpoint loading, peak generation
resources, and artifact footprint are reported separately and are never folded
silently into TTFT or TTC. Evidence is labeled **smoke** for execution-only
checks, **canary** for paired reproducible engineering evidence without a public
claim, or **publication** only after the sample, scorer, provenance, statistics,
cache-state, and publication-gate requirements in the evidence policy pass.

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
| Hardware | AWS g6/L4, `g6.8xlarge` |
| Request parallelism | 4 requests in flight |
| Output length for latency | Forced 256-token decode with `max_tokens=256` and `ignore_eos=true` |
| Latency repeats | 32 repeats per prepared input (8 prepared inputs → 256 measurements per method/context cell) |
| Latency input context lengths | 8k, 16k, and 32k prepared prompts, each assembled from distinct 2k-token documents (8k = 4 docs, 16k = 8 docs, 32k = 16 docs) |
| Latency document distinctness | Two examples from each dataset (8 total) are round-robin interleaved; documents are distinct within a request and across concurrent requests, while per-request `cache_salt` isolation and OS page-cache eviction keep repeated Vanilla hydrates cold |
| Latency job isolation | Each (method × context) row runs as its own single-node `g6.8xlarge` job (separate cluster + vLLM server) so no configuration contaminates another |
| Default Cachet method | Vanilla external KV (independent pre-RoPE documents; absolute-position injection) |
| Default document KV precision | Q8, represented as `fp8_e5m2` payloads |
| vLLM runtime KV dtype | `fp8_e5m2` |
| Cache residency | Local disk handoff bundles; Cachet rows hydrate document KV from disk during measured requests |
| Prefix-cache policy | vLLM prefix caching enabled with per-request `cache_salt` isolation for latency rows |
| GPU memory setting | `gpu_memory_utilization=0.90` at 8k/16k and `0.70` at 32k; matched within each method pair, not across context lengths |
| Runtime KV ownership | Shared GPU KV for the loaded document/system prefix during each request; private KV for request-specific prompt suffix and generated tokens |
| Score datasets | Biography, HotpotQA, MusiQue, NIAH (LongBench v2 and RULER are reserved columns; their staging/eval harness is not implemented yet) |
| Score metric | Full-dataset task score; explicitly `N/A` because full-dataset runs have not been conducted |
| Current descriptive evidence | [`appendix/main-vanilla-descriptive-evidence/`](appendix/main-vanilla-descriptive-evidence/) |
| Warm-prefix canary evidence | [`appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/) (historical A10G) |

The appendix includes prepared-suite warm-prefix smoke checks. Those checks are
not cold-hydrate latency rows, are not full benchmark scores, and are not
copied into the main score table.

## Main Latency And Resource Table

Latency values are seconds.

| Method | Input context | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Max Serving Concurrency | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8k | 4.3675 | 5.0451 | 27.6693 | 27.7153 | 11.0841 | 29.00x | 19.90 GiB |
| Baseline | 16k | 10.6750 | 11.3986 | 50.2451 | 50.3169 | 6.5108 | 14.50x | 19.90 GiB |
| Baseline | 32k | 33.5848 | 33.8648 | 123.7699 | 123.9187 | 2.8415 | 5.29x | 15.48 GiB |
| Vanilla&nbsp;KV | 8k | 6.2500 | 6.3029 | 22.7391 | 22.8147 | 15.5231 | 29.00x | 20.34 GiB |
| Vanilla&nbsp;KV | 16k | 12.5511 | 12.6988 | 32.2173 | 32.3743 | 13.0289 | 14.50x | 21.06 GiB |
| Vanilla&nbsp;KV | 32k | 24.8272 | 25.0384 | 50.1048 | 50.2997 | 10.1354 | 5.29x | 18.07 GiB |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | 8k | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | 16k | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | 32k | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) |
| CacheBlend | 8k | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) |
| CacheBlend | 16k | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) |
| CacheBlend | 32k | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) |
| InfoFlow&nbsp;KV | 8k | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) |
| InfoFlow&nbsp;KV | 16k | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) |
| InfoFlow&nbsp;KV | 32k | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) | N/A (not implemented) |

`Baseline` means vLLM receives the complete prompt and computes KV for
the system prompt, documents, user question, and generated tokens at request
time. `Vanilla KV` means the request still carries the full logical token
sequence, but Cachet's `DocumentKVConnector` reports the reusable
system/document prefix as already computed and hydrates those pages into
vLLM-managed GPU KV state from the persisted handoff bundle on local disk, so
vLLM skips prefilling the prefix and only prefills the request-specific suffix
plus generated tokens. The connector must report a matched prefix that is a
strict prefix of the request's own tokens (vLLM's V1 scheduler asserts
`num_computed_tokens <= request.num_tokens`); a request that carries only the
suffix text cannot also load the longer cached prefix, so the
suffix-only client mode is unsupported and is not used by these rows. `KV
Packet`, `CacheBlend`, and `InfoFlow KV` are planned methods and are not
implemented yet; their rows are placeholders. See the "Methods and
pre-computation" section below for each method's arm, KV pre-computation, and
serving connector.

Latency rows are generated with `request_parallelism=4`: the benchmark runner
issues up to four concurrent requests while collecting request-level TTFT and
TTC measurements. Each request's prompt is assembled from distinct 2k-token
documents (8k = 4, 16k = 8, 32k = 16), and requests are round-robin interleaved
across eight prepared inputs (two each from Biography, HotpotQA, MusiQue, and
NIAH) so concurrent in-flight requests hydrate distinct documents. Each
(method × context) cell runs as its own isolated single-node `g6.8xlarge` job
with 32 repeats per prepared input, yielding 256 successful
request-level measurements per cell. Cold-hydrate rows use per-request
`cache_salt` isolation plus OS page-cache eviction so repeated examples neither
reuse vLLM prefix-cache blocks nor read a warm page cache across measured
requests.

`cache_salt` is the namespace vLLM includes in its prefix-cache key. A static
salt lets identical prefixes share already-resident KV blocks across requests;
that is useful for warm-prefix ablations but does not measure disk-to-GPU
hydrate cost. The main table uses per-request salt values, so vLLM prefix
caching remains enabled but cannot turn repeated measurements into warm prefix
hits.

`P50 tok/s` is per-request decode throughput, not aggregate server throughput.
It is computed for each completed request as
`completion_tokens / (TTC - TTFT)`, then summarized across request-level
measurements. `Max Serving Concurrency` is the vLLM startup KV-pool capacity
estimate for the nominal input context, reported as an `x` multiplier. The 8k
and 16k Q4/Q8 jobs use `gpu_memory_utilization=0.90` and report 237,584 GPU
KV-cache tokens, or 29.00x and 14.50x. The internally matched 32k pair uses
`gpu_memory_utilization=0.70` and reports 173,392 tokens, or 5.29x. The 32k
memory setting differs from the shorter contexts, so cross-context trends are
descriptive. This field is not the request parallelism used for latency
measurement and is not copied from pressure-probe batch sizes.

`Peak GPU memory` is populated only from sampled runtime telemetry such as
`nvidia-smi` peak process/device memory during the benchmark run.

The current rows bind content-addressed inputs, source and wheel identity, exact
method semantics, and sanitized aggregate evidence. They remain descriptive:
the generic raw Baseline gate reports `benchmark does not contain a cache arm`
and `resource arm 'baseline_prefill' has no resource measurements`. That is a
schema/isolated-job qualification failure, not a claim that the measurements
are canonical canary evidence.

## Methods and pre-computation

Each method in the tables above maps to a specific KV pre-computation and serving
path. Methods differ not only in how KV is served but in how it is
**pre-computed**, so a new method is a correspondence between an arm, a
pre-computation routine, and a serving connector. This correspondence is recorded
in code as the source of truth in
[`document_kv_cache.methods`](../src/document_kv_cache/methods.py) (`MethodSpec` /
`METHOD_SPECS`); new methods declare their contract there.

- **Baseline** - no cache; vLLM recomputes all KV at request time (full prefill). Arm `baseline_prefill`.
- **Vanilla KV** (implemented) - each document's K/V is computed
  independently; keys are stored **pre-RoPE** (after QK normalization), documents
  are assembled in logical order, and the connector applies every key's true
  absolute position during injection. Values are unchanged and there is no
  cross-document recomputation. Multi-document quality can still be limited by
  missing cross-document attention. Arm `document_kv_cache`.
- **KV Packet** (planned) - no executable Cachet path is present. The upstream
  implementation must first be pinned and reproduced before its artifact and
  serving contracts are defined here. Q4 compression is orthogonal to the
  method.
- **CacheBlend** (planned) - builds on the same position-independent pre-RoPE
  foundation **and** recomputes a selected fraction of cross-document tokens with
  full context. The selective-recompute step is not implemented.
- **InfoFlow KV** (planned) - recovers cross-document information flow over reused KV; expected to build on the same position-independent pre-RoPE foundation. Routine not yet defined.

In short: Vanilla owns the reusable pre-RoPE generation and absolute-position
assembly path. CacheBlend and InfoFlow KV may share that foundation, but their
method-specific selection/recomputation logic is still absent. KV Packet's
Cachet pre-computation contract remains intentionally unspecified pending
upstream reproduction.

## Benchmark Dataset Score Table

No full-dataset score run was conducted. Every cell below is therefore explicit
`N/A`; the small matched score diagnostic that follows is not substituted into
this table.

| Method | Biography score | HotpotQA score | MusiQue score | NIAH score | LongBench v2 score | RULER score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | N/A (full dataset not run) | N/A (full dataset not run) | N/A (full dataset not run) | N/A (full dataset not run) | N/A (runner not implemented) | N/A (runner not implemented) |
| Vanilla&nbsp;KV | N/A (full dataset not run) | N/A (full dataset not run) | N/A (full dataset not run) | N/A (full dataset not run) | N/A (runner not implemented) | N/A (runner not implemented) |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | N/A (method not implemented) | N/A (method not implemented) | N/A (method not implemented) | N/A (method not implemented) | N/A (method/runner not implemented) | N/A (method/runner not implemented) |
| CacheBlend | N/A (method not implemented) | N/A (method not implemented) | N/A (method not implemented) | N/A (method not implemented) | N/A (method/runner not implemented) | N/A (method/runner not implemented) |
| InfoFlow&nbsp;KV | N/A (method not implemented) | N/A (method not implemented) | N/A (method not implemented) | N/A (method not implemented) | N/A (method/runner not implemented) | N/A (method/runner not implemented) |

Scores use each dataset's declared scorer over paired, real full-document
dataset samples;
**answer-found rate** remains the diagnostic span-containment metric where the
dataset contract declares it. Input length remains a latency stress dimension,
not a separate score column.
LongBench v2 and RULER are `N/A` because their staging/evaluation contracts
are not implemented. Pre-RoPE absolute-position correction removes the known
positional inconsistency between independently generated documents; it does not
restore attention between documents that were computed independently. Paired
multi-document quality evidence is therefore still required, and selective
cross-document recomputation remains a separate planned CacheBlend capability.

### Five-example score diagnostic

This matched diagnostic uses five content-addressed 8k examples per dataset,
20 requests per arm, request parallelism 4, and each dataset's declared primary
metric. It is descriptive/nonpublication evidence; the sample is too small for
a full-dataset or superiority claim.

| Method | Biography answer-found | HotpotQA answer F1 | MusiQue answer-found | NIAH exact match |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1.000000 | 0.108975 | 0.200000 | 0.000000 |
| Vanilla&nbsp;KV | 1.000000 | 0.040827 | 0.000000 | 0.000000 |

Exact unrounded means and the matched proof identity are in the
[sanitized evidence record](appendix/main-vanilla-descriptive-evidence/evidence.json).

## Document KV Precision Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, vLLM `0.23.0`,
`g6.8xlarge` (L4), 16k input context, 4 requests in flight, forced 256-token
decode, local disk handoff bundles, cold disk-to-GPU hydrate, per-request
`cache_salt` isolation. Both measured arms use
`gpu_memory_utilization=0.85`.

This is a coupled end-to-end precision comparison: the persisted document KV
payload **and** vLLM runtime KV dtype change together between Q8 and bf16. It is
not a payload-only ablation. Native packed-Q4 KV is unsupported.

| Document KV payload | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Answer-found / strict EM | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bf16 | 24.1012 | 24.2734 | 47.7200 | 47.9258 | 10.8321 | N/A (latency/resource scope only) | 17.97 GiB | 6.76x | 21.06 GiB | 8.01 / 28.88 GiB | bf16 document payload and bf16 runtime KV; descriptive/nonpublication |
| Q8 (`fp8_e5m2`) | 12.5363 | 12.6601 | 32.3071 | 32.4945 | 12.9445 | N/A (latency/resource scope only) | 8.98 GiB | 13.52x | 19.94 GiB | 5.79 / 28.52 GiB | Q8 document payload and Q8 runtime KV; descriptive/nonpublication |
| Q4 packed document KV | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Not implemented; requires a packed-Q4 payload layout and provider dequant or native serving-engine Q4 KV support. This compression can be combined with methods such as KV Packet but is not KV Packet itself. |

Each measured precision row uses eight inputs, 32 repeats per input, request
parallelism 4, and 256 successful request measurements. `Answer-found / strict
EM` is `N/A` because these isolated jobs declared latency/resource scope only;
the separate score diagnostic above is not a precision experiment.
Across the ablation tables, cache footprints and memory values use binary GiB;
`CPU RSS / host RAM` reports sampled peak process-tree RSS followed by sampled
peak host memory in use.

## Storage Tier Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, vLLM
`0.23.0`, `g6.8xlarge` (L4), 16k input context, 4 requests in flight, forced
256-token decode, `gpu_memory_utilization=0.90`, and cold disk-to-GPU hydrate
unless the storage tier itself is RAM.

| Storage tier | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RAM | 4.0717 | 4.0941 | 24.0191 | 24.0452 | 12.8351 | 8.98 GiB | 14.50x | 21.06 GiB | 12.54 / 35.16 GiB | 16 GiB prewarmed payload cache; 256/256 measured hits and zero measured backend reads |
| Disk | 12.6404 | 12.7685 | 32.1144 | 32.2573 | 13.1519 | 8.98 GiB | 14.50x | 21.06 GiB | 5.80 / 27.38 GiB | Default local-NVMe Vanilla path; OS page-cache eviction and 256 cold-load attestations |
| Unity Catalog | 18.5979 | 21.0534 | 38.2596 | 40.6724 | 13.0326 | 8.98 GiB | 14.50x | 21.06 GiB | 5.79 / 29.89 GiB | Mounted UC path; OS eviction succeeded, but backend cache state is unproven, so this is not strict cold-UC evidence |
| Hybrid RAM / disk / Unity Catalog | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Combined serving policy not run under this protocol |

## Hardware Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, vLLM
`0.23.0`, 16k input context, disk cache, 4 requests in flight, forced
256-token decode, and `gpu_memory_utilization=0.90`.

| Hardware | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AWS g6/L4, `g6.8xlarge` | 12.5511 | 12.6988 | 32.2173 | 32.3743 | 13.0289 | 8.98 GiB | 14.50x | 21.06 GiB | 5.80 / 27.70 GiB | Descriptive/nonpublication; two local 450 GB disks |
| AWS g5/A10G, `g5.8xlarge` | 10.7994 | 10.9186 | 23.8841 | 24.0102 | 19.5611 | 8.98 GiB | 14.51x | 21.16 GiB | 5.82 / 29.50 GiB | Descriptive setting comparison; one local 900 GB disk, so storage topology can contribute |

Because the local-disk topology changes with the GPU/node type, this is a
descriptive setting comparison rather than a GPU-only one-variable ablation.

## Serving Platform Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV,
`g6.8xlarge` (L4), 16k input context, disk cache, 4 requests in flight, forced
256-token decode, and `gpu_memory_utilization=0.90`.

| Serving platform | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM | 12.5511 | 12.6988 | 32.2173 | 32.3743 | 13.0289 | 8.98 GiB | 14.50x | 21.06 GiB | 5.80 / 27.70 GiB | Current descriptive Vanilla result |
| SGLang | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Q8 pre-RoPE serving path is not implemented |

## Directory Layout

| Folder | Purpose |
| --- | --- |
| [`appendix/main-vanilla-descriptive-evidence/`](appendix/main-vanilla-descriptive-evidence/) | Current compact Q4/Q8 Baseline-versus-Vanilla, precision, storage, hardware, platform, and score-diagnostic evidence; descriptive/nonpublication-qualified |
| [`appendix/representative-bf16-qwen3-4b-canaries/`](appendix/representative-bf16-qwen3-4b-canaries/) | Current sanitized Vanilla BF16 canaries and matched direct-versus-legacy cold-load ablation; non-publication-qualified and not copied into the main tables |
| [`appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/) | Historical A10G warm-prefix canary evidence and Databricks provenance (predates the current g6/L4 protocol; folder name retained because it is referenced by committed release-evidence records) |
| [`databricks/`](databricks/) | Notes for Databricks provenance; historical committed mirrors have been removed |
| [`_template/`](_template/) | Required table shape for future public benchmark result folders |

Do not add raw Databricks Jobs API responses, credentials, package wheels,
driver logs, generated datasets, prompt payload blobs, or local scratch output
to this directory.
