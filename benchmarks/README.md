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

Blank numeric cells mean the row has not been measured, or the run has not
completed, under the current protocol yet. A blank cell is not a zero.

> **Evidence status: provisional / non-publication-qualified.** The populated
> numbers below are retained as engineering observations, but their detailed
> measurements are currently referenced only through private DBFS paths. The
> score arms also use different sample counts. Until matched sanitized evidence
> and a passing publication gate are committed, these rows must not be cited as
> released Cachet performance or quality claims. See the
> [evidence policy](../docs/evidence-policy.md).

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
| Latency repeats | 64 repeats per prepared input (4 prepared inputs → 256 measurements per method/context cell) |
| Latency input context lengths | 8k, 16k, and 32k prepared prompts, each assembled from distinct 2k-token documents (8k = 4 docs, 16k = 8 docs, 32k = 16 docs) |
| Latency document distinctness | Documents are distinct within a request and across the 4 concurrent requests in each wave (round-robin example interleaving); the small document pool repeats across the 64 repeats, with per-request `cache_salt` isolation and OS page-cache eviction keeping every hydrate cold |
| Latency job isolation | Each (method × context) row runs as its own single-node `g6.8xlarge` job (separate cluster + vLLM server) so no configuration contaminates another |
| Default Cachet method | Vanilla external KV |
| Default document KV precision | Q8, represented as `fp8_e5m2` payloads |
| vLLM runtime KV dtype | `fp8_e5m2` |
| Cache residency | Local disk handoff bundles; Cachet rows hydrate document KV from disk during measured requests |
| Prefix-cache policy | vLLM prefix caching enabled with per-request `cache_salt` isolation for latency rows |
| Runtime KV ownership | Shared GPU KV for the loaded document/system prefix during each request; private KV for request-specific prompt suffix and generated tokens |
| Score datasets | Biography, HotpotQA, MusiQue, NIAH (LongBench v2 and RULER are reserved columns; their staging/eval harness is not implemented yet) |
| Score metric | Full-dataset task score; blank until full-dataset runs complete |
| Warm-prefix canary evidence | [`appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/) (historical A10G) |

The appendix includes prepared-suite warm-prefix smoke checks. Those checks are
not cold-hydrate latency rows, are not full benchmark scores, and are not
copied into the main score table.

## Main Latency And Resource Table

Latency values are seconds.

| Method | Input context | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Max Serving Concurrency | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8k | 5.000 | 5.152 | 27.713 | 27.986 | 11.266 | 29.00x | 19.901 GiB |
| Baseline | 16k | 11.849 | 12.322 | 52.800 | 53.112 | 6.278 | 14.50x | 19.901 GiB |
| Baseline | 32k | 35.923 | 36.516 | 135.326 | 137.445 | 2.580 | 7.25x | 19.901 GiB |
| vanilla&nbsp;KV | 8k | 1.806 | 1.850 | 18.765 | 18.888 | 15.090 | 29.00x | 20.851 GiB |
| vanilla&nbsp;KV | 16k | 3.980 | 4.198 | 24.177 | 24.435 | 12.681 | 14.50x | 20.849 GiB |
| vanilla&nbsp;KV | 32k | 7.772 | 8.081 | 33.590 | 33.904 | 9.919 | 7.25x | 22.028 GiB |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | 8k |  |  |  |  |  |  |  |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | 16k |  |  |  |  |  |  |  |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | 32k |  |  |  |  |  |  |  |
| CacheBlend | 8k |  |  |  |  |  |  |  |
| CacheBlend | 16k |  |  |  |  |  |  |  |
| CacheBlend | 32k |  |  |  |  |  |  |  |
| InfoFlow&nbsp;KV | 8k |  |  |  |  |  |  |  |
| InfoFlow&nbsp;KV | 16k |  |  |  |  |  |  |  |
| InfoFlow&nbsp;KV | 32k |  |  |  |  |  |  |  |

`Baseline` means vLLM receives the complete prompt and computes KV for
the system prompt, documents, user question, and generated tokens at request
time. `vanilla KV` means the request still carries the full logical token
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
across the four prepared inputs (one per dataset: Biography, HotpotQA, MusiQue,
NIAH) so the four concurrent in-flight requests always hydrate distinct
documents. Each (method × context) cell runs as its own isolated single-node
`g6.8xlarge` job with 64 repeats per prepared input, yielding 256 successful
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
estimate for the nominal input context, reported as an `x` multiplier. The
current Q4/Q8 vLLM logs on `g6.8xlarge`/L4 report 237,584 GPU KV-cache tokens,
which corresponds to 29.00x at 8k, 14.50x at 16k, and 7.25x at 32k. It is not
the request parallelism used for latency measurement and is not copied from
pressure-probe batch sizes.

`Peak GPU memory` is populated only from sampled runtime telemetry such as
`nvidia-smi` peak process/device memory during the benchmark run.

The provisional populated rows come from the private
`dbfs:/benchmarks/cachet/lat4-20260708/runs/` jobs (256 successful request-level
measurements per cell, zero errors). The private paths preserve workspace
provenance but are not sanitized committed publication evidence. Each
request carries the full logical prompt plus a `DocumentKVConnector` that marks
the document/system prefix as already computed and hydrates those pages into
vLLM-managed GPU KV state from the persisted per-document handoff bundles on
local disk. The connector defensively caps the matched prefix to the visible
request length, so a misconfigured request can never violate vLLM's V1 scheduler
assertion (`num_computed_tokens <= request.num_tokens`) and crash EngineCore.

The provisional engineering observation is that on `g6.8xlarge`/L4 at
`request_parallelism=4` with distinct multi-document contexts, vanilla external
KV **reduced** first-token latency versus baseline at every measured context,
and the advantage grows with
context length: P50 TTFT drops from 5.000 s to 1.806 s at 8k (2.8×), from
11.849 s to 3.980 s at 16k (3.0×), and from 35.923 s to 7.772 s at 32k (4.6×),
because hydrating the cached document/system prefix from local disk is cheaper
than recomputing it and frees the GPU for decode. This reverses the earlier
A10G / `request_parallelism=8` single-document result (retained in git history),
where cold hydrate was *slower* than prefill at 8k and 16k. The downstream
benefit persists across the board: vanilla KV also delivers higher per-request
decode throughput (8k: 15.1 vs 11.3 tok/s; 16k: 12.7 vs 6.3 tok/s; 32k: 9.9 vs
2.6 tok/s) and lower total time-to-completion (8k: 18.8 s vs 27.7 s; 16k: 24.2 s
vs 52.8 s; 32k: 33.6 s vs 135.3 s).

## Methods and pre-computation

Each method in the tables above maps to a specific KV pre-computation and serving
path. Methods differ not only in how KV is served but in how it is
**pre-computed**, so a new method is a correspondence between an arm, a
pre-computation routine, and a serving connector. This correspondence is recorded
in code as the source of truth in
[`document_kv_cache.methods`](../src/document_kv_cache/methods.py) (`MethodSpec` /
`METHOD_SPECS`); new methods declare their contract there.

- **Baseline** - no cache; vLLM recomputes all KV at request time (full prefill). Arm `baseline_prefill`.
- **vanilla KV** (implemented) - per-document KV computed independently, stored **post-RoPE**, hydrated into GPU KV by the `cachet` connector; no cross-chunk recomputation. Correct for single-document / true-prefix reuse; multi-document quality is limited by missing cross-document attention. Arm `document_kv_cache`.
- **KV Packet** (planned) - no executable Cachet path is present. The upstream
  implementation must first be pinned and reproduced before its artifact and
  serving contracts are defined here. Q4 compression is orthogonal to the
  method.
- **CacheBlend** (planned) - stores **position-independent pre-RoPE keys** (re-roped to their true offset at injection; foundation implemented, flag-gated via `CACHET_TRANSFORMERS_PRE_ROPE`) **and** recomputes a small fraction of cross-chunk tokens with full context to recover multi-document quality. The selective-recompute step is not yet implemented.
- **InfoFlow KV** (planned) - recovers cross-document information flow over reused KV; expected to build on the same position-independent pre-RoPE foundation. Routine not yet defined.

In short: vanilla KV uses post-RoPE pre-computation; CacheBlend and InfoFlow KV
require position-independent pre-RoPE pre-computation. KV Packet's Cachet
pre-computation contract remains intentionally unspecified pending upstream
reproduction.

## Benchmark Dataset Score Table

**Provisional comparison:** the raw scores are retained, but Baseline used 200
examples per dataset while vanilla KV used 50. The rows are not a matched,
paired comparison and their detailed evidence is available only at private
DBFS paths. They are therefore non-publication-qualified and cannot establish
quality parity or a statistically supported quality delta.

| Method | Biography score | HotpotQA score | MusiQue score | NIAH score | LongBench v2 score | RULER score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.730 | 0.840 | 0.505 | 0.995 |  |  |
| vanilla&nbsp;KV | 0.680 | 0.300 | 0.000 | 1.000 |  |  |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) |  |  |  |  |  |  |
| CacheBlend |  |  |  |  |  |  |
| InfoFlow&nbsp;KV |  |  |  |  |  |  |

Scores are the diagnostic **answer-found rate** — the fraction of examples
whose gold answer appears in the generated output — over real full-document
dataset samples on
`g6.8xlarge`/L4 (Qwen3-4B-Instruct, 4-bit weights, `fp8_e5m2` runtime KV). Exact
match is ≈0.00 across all datasets because the instruct model returns verbose,
explanatory answers rather than the bare gold span, so answer-found is the
meaningful task signal. The score table has no input-context column because
input-context length is a latency stress dimension, not a separate scoring
condition. The Baseline row is measured over 200 samples per dataset
(`dbfs:/benchmarks/cachet/lat4-20260708/runs/score-baseline/`, zero request
errors). LongBench v2 and RULER are reserved columns whose staging/eval harness
is not implemented yet, so they remain blank. The vanilla KV row is measured over
50 samples per dataset (`dbfs:/benchmarks/cachet/lat4-20260708/runs/score-cachet-50/`,
zero request errors); a reduced count was used because per-document KV handoff
generation over real multi-document examples (MuSiQue carries 20 documents per
example) OOMs the host at 200 samples per dataset.

The provisional score table flags a quality risk the latency table cannot:
vanilla external KV is close to the unmatched Baseline row on the two
single-document diagnostics (Biography 0.68 vs 0.73, NIAH 1.00 vs 1.00), while
the unmatched multi-document diagnostics are substantially lower (HotpotQA
0.30 vs 0.84 with 10 documents; MuSiQue 0.00 vs 0.505 with 20 documents).
Because sample identities and counts are not matched, this is a hypothesis to
verify with paired evidence rather than a parity or degradation claim. The
suspected mechanism is the cross-chunk positional-consistency problem: each
document's KV is materialized independently
with its cached positions starting at 0, so concatenating several documents into
one prefix yields positionally inconsistent KV that vanilla reuse does not
correct. Recovering multi-document quality requires two things: (1) RoPE
re-alignment across chunks, and (2) selective recomputation of a subset of
cross-chunk tokens with full context (the mechanism CacheBlend/LMCache
implement). A flag-gated pre-RoPE re-alignment foundation exists in the code
(`CACHET_TRANSFORMERS_PRE_ROPE`; stores position-independent keys re-roped at
their true offset during injection), and it is off by default so these vanilla
KV numbers are post-RoPE. Empirically, re-alignment alone recovers single-document
positional correctness but does **not** recover multi-document quality (measured:
HotpotQA and MuSiQue stay within noise of the numbers above) — the missing piece
is the selective recomputation of cross-document attention, which is the planned
`CacheBlend` method (see "Methods and pre-computation"). The fast multi-document
TTFT in the latency table above is therefore quality-preserving today only for
single-document (or true-prefix) reuse.

## Document KV Precision Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, vLLM `0.23.0`,
`g6.8xlarge` (L4), 16k input context, 4 requests in flight, forced 256-token
decode, local disk handoff bundles, cold disk-to-GPU hydrate, per-request
`cache_salt` isolation, and vLLM runtime KV dtype `fp8_e5m2`.

This ablation varies the document KV payload stored on disk. GPU KV residency
is still governed by the vLLM runtime KV dtype unless the serving engine gains
native packed-Q4 KV pages.

| Document KV payload | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Answer-found / strict EM | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bf16 |  |  |  |  |  |  |  |  |  |  | Not yet re-measured under the current protocol |
| Q8 (`fp8_e5m2`) |  |  |  |  |  |  |  |  |  |  | Default document KV precision; not yet re-measured under the current protocol |
| Q4 packed document KV |  |  |  |  |  |  |  |  |  |  | Implementation pending; requires a packed-Q4 payload layout and provider dequant or native serving-engine Q4 KV support. This compression can be combined with methods such as KV Packet but is not KV Packet itself. |

These rows are not yet re-measured under the current protocol (g6/L4, request
parallelism 4, N x 2k distinct documents, 64 repeats). The current L4 startup
logs report 237,584 GPU KV-cache tokens (29.00x/14.50x/7.25x at 8k/16k/32k); the
older A10G warm-prefix canary figures (237,728 tokens, 4.92 GB Q8 / 9.83 GB bf16
footprints) are retained only in the historical appendix.

## Storage Tier Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, vLLM
`0.23.0`, `g6.8xlarge` (L4), 16k input context, 4 requests in flight, forced
256-token decode, and cold disk-to-GPU hydrate unless the storage tier itself
is RAM.

| Storage tier | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RAM |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |
| Disk |  |  |  |  |  |  |  |  |  | Current default for vanilla KV; not yet re-measured under the current protocol |
| Unity Catalog |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |
| Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Hardware Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV, vLLM
`0.23.0`, 16k input context, disk cache, 4 requests in flight, forced
256-token decode.

| Hardware | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AWS g6/L4, `g6.8xlarge` |  |  |  |  |  |  | 14.50x |  |  | Current default; latency measured in the Main Latency And Resource Table above (per-context TTFT/TTC/tok/s not duplicated here) |
| AWS g5/A10G, `g5.8xlarge` |  |  |  |  |  |  |  |  |  | Historical (old warm-prefix protocol; see the historical appendix); not re-measured |

## Serving Platform Ablation

Configuration: Qwen3-4B-Instruct, 4-bit model weights, Q8 document KV,
`g6.8xlarge` (L4), 16k input context, disk cache, 4 requests in flight, forced
256-token decode.

| Serving platform | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Cache footprint | Max Serving Concurrency | Peak GPU memory | CPU RSS / host RAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM |  |  |  |  |  |  | 14.50x |  |  | Current default; latency measured in the Main Latency And Resource Table above |
| SGLang |  |  |  |  |  |  |  |  |  | Not measured under the current protocol |

## Directory Layout

| Folder | Purpose |
| --- | --- |
| [`appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/) | Historical A10G warm-prefix canary evidence and Databricks provenance (predates the current g6/L4 protocol; folder name retained because it is referenced by committed release-evidence records) |
| [`databricks/`](databricks/) | Notes for Databricks provenance; historical committed mirrors have been removed |
| [`_template/`](_template/) | Required table shape for future public benchmark result folders |

Do not add raw Databricks Jobs API responses, credentials, package wheels,
driver logs, generated datasets, prompt payload blobs, or local scratch output
to this directory.
