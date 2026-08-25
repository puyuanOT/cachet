# Cachet Benchmarks

This directory is the public benchmark surface for Cachet.

> **Status: vLLM 0.27.1 campaign pending.** No latency, resource, ablation, or
> full-dataset score produced by the reset campaign has been published yet.
> Every metric cell below is therefore `N/A (0.27.1 campaign pending)` unless
> the method or runner is unsupported. `N/A` is never a zero.

Superseded serving-engine evidence and its numeric tables were removed instead
of being mixed with the reset. A result may replace a pending cell only after
its exact source, wheel, runtime lock, inputs, hardware qualification,
Databricks execution attestation, and publication-gate record are committed.

## Frozen Campaign Configuration

The closed design is implemented by
[`document_kv_cache.publication_campaign`](../src/document_kv_cache/publication_campaign.py).
The source snapshot and generated campaign record must be frozen before any
production job is submitted.

| Field | Frozen value |
| --- | --- |
| Campaign | `vllm-0271-publication-v1` |
| Model | `Qwen/Qwen3-4B-Instruct-2507`, served as `qwen3:4b-instruct` |
| Model weights | bitsandbytes 4-bit runtime weights; the model identifier is not itself a prequantized 4-bit checkpoint |
| Serving engine | vLLM `0.27.1` |
| Core serving hardware | AWS g6/L4, `g6.8xlarge` |
| Handoff generation hardware | Qualified AWS g6e/L40S producers only; generated bundles are reused by timed serving jobs |
| Default document/runtime KV | Q8, `fp8_e5m2`, pre-RoPE keys with absolute-position injection |
| Input contexts | 8k, 16k, and 32k prepared prompts |
| Closed-loop request concurrency | 1, 2, and 4; zero think time |
| Dataset coverage | Biography, HotpotQA, MusiQue, and NIAH |
| Examples | 32 distinct examples per dataset |
| Repeats | 2 per example within each deployment block |
| Deployment blocks | 5 matched fresh-cluster blocks |
| Requests per method/context/concurrency cell and block | 256 |
| Decode | Forced 256-token latency decode; exact decoding pins come from the closed execution record |
| No-retry job timeouts | 8k c1/c2/c4: 6/4/4h; 16k: 8/6/4h; 32k: 12/8/4h; all c4 auxiliary jobs: 4h |
| Experimental units | Core Baseline/Vanilla pair; Disk/RAM/UC trio; 16k-c4 core pair plus BF16/A10G four-job wave |
| Analysis | Paired hierarchical bootstrap over deployment blocks and examples; no post-hoc cell significance |
| Full-score scope | One complete paired pass over every selected dataset row; no padding, truncation, sampling replacement, or answer-quality preservation gate |
| Budget | 1,024 aggregate GPU-hours, 900 active reserved hours, 124 hours unreserved headroom, at most 16 parallel jobs |
| Retained ledger opening | 56.593291 reconciled GPU-hours; exact 152/14/152 post-migration append-only prefix is campaign-bound |
| Frozen generation workload | 72,871,510 cache-prefix tokens across Q8 latency, BF16 latency, and complete full-score handoffs; 578.345317 GPU-hours at the 35 token/GPU-s gate |

`Baseline` sends the complete logical prompt to vLLM and computes all KV at
request time. `Vanilla KV` reuses independently generated document KV, stores
keys pre-RoPE, assembles documents in logical order, applies absolute positions
at injection, and measures storage-to-GPU hydration inside TTFT. Vanilla is not
required to preserve Baseline answer quality: the full evaluation is intended
to measure the quality loss, especially for multi-document QA.

KV Packet, CacheBlend, and InfoFlow KV have no executable Cachet benchmark path
in this campaign. Their cells remain explicitly unsupported rather than being
estimated from another method.

## Main Latency And Resource Table

Latency values will be reported in seconds. Each pending row represents five
matched deployment blocks; no prior measurement is carried forward.

| Method | Input context | Concurrency setting | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 decode tok/s | Configured closed-loop concurrency | Peak GPU process memory | Peak host memory | Peak process-tree RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8k | 1 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Vanilla&nbsp;KV | 8k | 1 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Baseline | 8k | 2 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Vanilla&nbsp;KV | 8k | 2 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Baseline | 8k | 4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Vanilla&nbsp;KV | 8k | 4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Baseline | 16k | 1 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Vanilla&nbsp;KV | 16k | 1 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Baseline | 16k | 2 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Vanilla&nbsp;KV | 16k | 2 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Baseline | 16k | 4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Vanilla&nbsp;KV | 16k | 4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Baseline | 32k | 1 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Vanilla&nbsp;KV | 32k | 1 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Baseline | 32k | 2 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Vanilla&nbsp;KV | 32k | 2 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Baseline | 32k | 4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |
| Vanilla&nbsp;KV | 32k | 4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) |

## Paired Latency Estimands

Each estimand is computed from five matched deployment blocks. A speedup is
`reference latency / treatment latency`, so values above 1 mean that the named
treatment is faster. The table reports estimation with pointwise 95% paired
hierarchical-bootstrap intervals; it is not a grid of post-hoc significance
tests.

| Treatment vs reference | Setting | TTFT geometric speedup | TTFT 95% CI | TTC geometric speedup | TTC 95% CI | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vanilla KV vs Baseline | 8k, concurrency 1 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| Vanilla KV vs Baseline | 8k, concurrency 2 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| Vanilla KV vs Baseline | 8k, concurrency 4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| Vanilla KV vs Baseline | 16k, concurrency 1 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| Vanilla KV vs Baseline | 16k, concurrency 2 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| Vanilla KV vs Baseline | 16k, concurrency 4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| Vanilla KV vs Baseline | 32k, concurrency 1 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| Vanilla KV vs Baseline | 32k, concurrency 2 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| Vanilla KV vs Baseline | 32k, concurrency 4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| BF16 payload/runtime KV vs Q8 | 16k, concurrency 4, L4/NVMe | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |
| RAM vs Disk | 16k, concurrency 4, L4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Dedicated five-block storage schedule pending |
| Unity Catalog vs Disk | 16k, concurrency 4, L4 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Dedicated five-block storage schedule pending |
| A10G vs L4 | 16k, concurrency 4, local NVMe | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Five matched blocks pending |

## Unsupported Method Status

| Method | Latency/resource result | Full-dataset score | Reason |
| --- | --- | --- | --- |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | N/A (method not implemented) | N/A (method not implemented) | No pinned executable integration or Cachet artifact/serving contract |
| CacheBlend | N/A (method not implemented) | N/A (method not implemented) | Selective cross-document recomputation is not implemented |
| InfoFlow&nbsp;KV | N/A (method not implemented) | N/A (method not implemented) | Cross-document information-flow method is not implemented |

## Benchmark Dataset Score Table

The reset requires a complete paired evaluation over every selected row in the
four implemented datasets. The earlier small diagnostic is not retained or
substituted for this table. Parser-status cells report the complete counts for
`ok`, `missing_block`, `multiple_or_malformed_blocks`, `extraneous_text`,
`nested_block`, and `empty_answer`, including explicit zeros for unobserved
states. Invalid parses score zero rather than disappearing from a denominator.

| Dataset | Official metric | Baseline | Baseline parser-status counts | Vanilla KV | Vanilla parser-status counts | Vanilla − Baseline | Paired 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Biography | Normalized-title exact match | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| HotpotQA | Answer exact match | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| HotpotQA | Answer F1 | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| MusiQue | Official answer exact match, alias-max | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| MusiQue | Official answer F1, alias-max | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| NIAH | Exact-value overall accuracy | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| LongBench v2 | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) |
| RULER | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) |

NIAH is additionally reported as the full 3-by-3 grid below; every cell is a
governed exact-value accuracy over its bound source examples.

| Context | Needle position | n | Baseline accuracy | Vanilla KV accuracy | Vanilla − Baseline | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8k | 10% | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| 8k | 50% | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| 8k | 90% | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| 16k | 10% | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| 16k | 50% | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| 16k | 90% | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| 32k | 10% | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| 32k | 50% | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |
| 32k | 90% | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) | N/A (0.27.1 full evaluation pending) |

## Ablation Tables

Every implemented ablation is refreshed on vLLM 0.27.1. Precision and hardware
reuse the exact 16k, concurrency-4 core Vanilla control from the same deployment
block. Storage uses a dedicated matched Disk/RAM/Unity Catalog trio per block:
two examples per dataset, 32 repeats, 256 requests, and concurrency 4.
Those two identities are the lowest domain-separated SHA-256 ranks within the
canonical 32-example dataset domain; callers cannot choose a favorable subset.

### Document KV Precision

| Document KV payload/runtime KV | P50 TTFT | P50 TTC | Status |
| --- | ---: | ---: | --- |
| Q8 / `fp8_e5m2` | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Core 16k, concurrency-4 Vanilla anchor |
| bf16 / bf16 | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Implemented refresh cell |
| Packed Q4 | N/A (not implemented) | N/A (not implemented) | No packed-Q4 payload and serving dequantization contract |

### Storage Tier

| Storage tier | P50 TTFT | P50 TTC | Status |
| --- | ---: | ---: | --- |
| Local NVMe disk | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Dedicated strict-cold storage control in each block |
| RAM | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | 16-GiB provider payload cache; eight targets populated and verified before 256 measured hits |
| Unity Catalog mounted path | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Dedicated mounted-path cell; OS eviction and exact bytes are proved while backend-cache state remains unproven |
| Hybrid RAM/disk/Unity Catalog | N/A (not implemented) | N/A (not implemented) | Combined serving policy is unsupported |

### Hardware

| Hardware | P50 TTFT | P50 TTC | Status |
| --- | ---: | ---: | --- |
| AWS g6/L4, `g6.8xlarge` | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Core 16k, concurrency-4 Vanilla anchor |
| AWS g5/A10G, `g5.8xlarge` | N/A (0.27.1 campaign pending) | N/A (0.27.1 campaign pending) | Implemented compatibility refresh cell; storage topology remains part of the setting |

### Serving Platform

| Serving platform | Result | Status |
| --- | --- | --- |
| vLLM 0.27.1 | N/A (0.27.1 campaign pending) | Campaign target |
| SGLang | N/A (Q8 pre-RoPE serving path not implemented) | Unsupported for the main campaign |

## Evidence And Publication Gate

The appendix is intentionally empty of result evidence while the campaign is
pending. Production evidence must be sanitized and content-addressed; raw Jobs
API responses, credentials, wheels, logs, generated datasets, prompt payloads,
and local scratch output do not belong here.

Publication requires all of the following:

- the frozen 0.27.1 source and wheel identity;
- one closed campaign record and exact input provenance;
- qualified L4/A10G/L40S runtime artifacts as applicable;
- 16 independent Q8 and 16 independent BF16 no-retry L40S
  handoff-generation attestations;
- 23 attempt-zero, no-retry CPU coordinator attestations (two Q8/BF16 tree
  closers, one latency-source closer, and 20 full-score ready/evidence closers),
  each proving zero GPU-ledger mutation;
- matched fresh-cluster latency execution records for all planned cells;
- the corrected complete paired score execution record;
- resource, cache-state, and Databricks control-plane attestations; and
- a passing publication-gate record.

## Directory Layout

| Folder | Purpose |
| --- | --- |
| [`appendix/`](appendix/) | Future sanitized 0.27.1 campaign evidence after the publication gate passes |
| [`databricks/`](databricks/) | Databricks evidence-handling policy; no run results are currently published |
| [`native-engine/`](native-engine/) | Native integration scope; not a latency result folder |
| [`sglang/`](sglang/) | Explicit unsupported/pending SGLang benchmark status |
| [`storage/`](storage/) | Storage-ablation campaign status |
| [`vllm/`](vllm/) | vLLM 0.27.1 campaign index |
| [`_template/`](_template/) | Required report shape for future sanitized evidence |
