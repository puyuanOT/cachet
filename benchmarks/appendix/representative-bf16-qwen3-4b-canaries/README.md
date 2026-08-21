# Representative BF16 Qwen3-4B Vanilla-v2 Canaries

This folder preserves sanitized engineering evidence for the registered
**Vanilla-v2** method. Vanilla v2 computes every document independently,
captures keys after QK normalization but before RoPE, assembles the documents
in logical order, and applies each key's true absolute assembled position at
injection. Values are unchanged. This fixes the positional inconsistency in the
superseded post-RoPE implementation; it does not recover cross-document
attention that was absent during independent generation.

These records are deliberately **non-publication-qualified**. They use BF16
model weights, BF16 runtime KV, BF16 document KV, two examples, three repeats,
and request parallelism 1. They therefore do not populate or revise any value
in the [main Q4-weight/Q8-document-KV tables](../../README.md), whose numeric
cells remain blank until that exact protocol is run.

## Bound Identity

| Field | Value |
| --- | --- |
| Source revision | `b4b142c79443fcca62b08044d0937298eab3f71d` |
| Cachet wheel SHA-256 | `5d91052aa5e92db64c3ba21924ae1805b7671c8c19bdc600fb477956dca78f90` |
| Model and tokenizer | `Qwen/Qwen3-4B-Instruct-2507` |
| Model and tokenizer revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Model weights / runtime KV / document KV | BF16 / BF16 / BF16; no quantization |
| Vanilla contract | method v2; `pre_rope`; `rerope_at_injection`; theta `5000000.0`; rotary dimension `128` |
| vLLM | `0.23.0` |
| SGLang | `0.5.10.post1` |

The canonical records retain exact package pins, hardware fingerprints,
logical-sample digests, physical-transform identities, decoding settings,
sanitized request correlations, and cold-load attestations. Raw Jobs API
responses, logs, payloads, prompt text, and generated helper summaries are not
committed.

Execution-source binding for every JSON record in this directory, including
SGLang, is an envelope property: the signed source commit and wheel hash above,
each canonical file hash below, and the committed verification tests. No
detached JSON is standalone source provenance. In particular, the generic vLLM
arm `source_revision` fields remain null; the records have not been altered
post hoc to manufacture that field.

## Evidence Files

| File | Scope | Canonical file SHA-256 |
| --- | --- | --- |
| [`g6-vllm-8k-64-three-arm-canary-v2.json`](g6-vllm-8k-64-three-arm-canary-v2.json) | L4, 8k input target, 64 output tokens | `5a8e869dbf75e9e6d380278c18d5b29e0ea7d4069c5e4f67b326ff179c393d2f` |
| [`g6-vllm-16k-256-three-arm-canary-v2.json`](g6-vllm-16k-256-three-arm-canary-v2.json) | L4, 16k input target, 256 output tokens | `c5b44cf9ab4fd40e2bbf829e1cc033c57e05c88f303e50c3706cda1a3eaeb32c` |
| [`g5-vllm-8k-64-three-arm-canary-v2.json`](g5-vllm-8k-64-three-arm-canary-v2.json) | A10G compatibility canary, 8k input target, 64 output tokens | `628781b3bbdf716e97c935987105897335c767f1e10f4b877fc9ec83c72bc630` |
| [`g6-sglang-4k-32-paired-smoke-evidence-v2.json`](g6-sglang-4k-32-paired-smoke-evidence-v2.json) | L4 native-handoff execution smoke | `c226d949e1e9e612ccd4aec34e0e9dc78f6541f111a51540ffacdee709387174` |
| [`vanilla-v2-cold-optimization.json`](vanilla-v2-cold-optimization.json) | Six-job direct-versus-legacy cold-load ablation | `c96c60dd5c9511a9c458d9ebf0f3903738a00dbeb848f434e9e69a86063ff112` |

## vLLM Three-Arm Canaries

Each arm ran in its own single-node job with zero warmups and the same
`2 examples × 3 repeats`. Six requests therefore means repeated measurements
of two HotpotQA examples, not six independent samples. The three arms are:

- `baseline_prefill`: standard no-cache full-prompt recomputation.
- `full_prefix_prefill`: an exact full-prefix, one-segment post-RoPE control.
- `vanilla_prefill`: Vanilla-v2 independent pre-RoPE document segments with
  true-position re-RoPE at injection.

Latency values below are seconds. Ratios are baseline P50 divided by arm P50;
a ratio below 1 means the cache arm was slower. The JSON records retain every
measurement, P95, paired statistic, and full-precision value.

| Hardware | Input / output | Arm | Requests | P50 TTFT | P50 TTC | Mean F1 | TTFT ratio | TTC ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `g6.8xlarge` / L4 | 8192 / 64 | baseline | 6 | 1.565836 | 3.934683 | 0.05625 | 1.0000 | 1.0000 |
| `g6.8xlarge` / L4 | 8192 / 64 | full-prefix control | 6 | 1.850524 | 4.211797 | 0.05625 | 0.8462 | 0.9342 |
| `g6.8xlarge` / L4 | 8192 / 64 | Vanilla v2 | 6 | 2.909791 | 5.274294 | 0.02381 | 0.5381 | 0.7460 |
| `g6.8xlarge` / L4 | 16384 / 256 | baseline | 6 | 3.859666 | 14.706137 | 0.01474 | 1.0000 | 1.0000 |
| `g6.8xlarge` / L4 | 16384 / 256 | full-prefix control | 6 | 3.245188 | 14.078988 | 0.01392 | 1.1894 | 1.0445 |
| `g6.8xlarge` / L4 | 16384 / 256 | Vanilla v2 | 6 | 5.846156 | 16.687317 | 0.00000 | 0.6602 | 0.8813 |
| `g5.8xlarge` / A10G | 8192 / 64 | baseline | 6 | 1.409549 | 2.760885 | 0.05625 | 1.0000 | 1.0000 |
| `g5.8xlarge` / A10G | 8192 / 64 | full-prefix control | 6 | 1.737752 | 3.089778 | 0.05530 | 0.8111 | 0.8936 |
| `g5.8xlarge` / A10G | 8192 / 64 | Vanilla v2 | 6 | 2.569946 | 3.921416 | 0.02381 | 0.5485 | 0.7041 |

All 54 requests completed without request errors and with exact server token
accounting. In each trio, the 12 cache measurements join one-to-one with 12
cold-read attestations: page-cache eviction succeeded, the payload cache was
disabled, and exactly one load succeeded. The cache arms loaded 8,144
block-aligned tokens at 8k and 16,336 at 16k. Across all three trios that is 36
cold-attested cache requests.

The 16k canary quality gate passes. Both 8k quality gates fail with the same
retained issue:

> hotpotqa:document_kv_cache:vanilla_prefill paired 'f1' lower bound -0.0625 exceeds allowed regression 0.02

The positional contract is now correct, but the two-example result does not
establish general quality. In particular, independent document generation
still omits cross-document attention. Every record's `v1_evidence.ok` also
remains false because a two-example HotpotQA canary is not the required
multi-dataset publication suite.

## Cold-Load Optimization

The vLLM provider now selects `direct_global_snapshot` automatically for a
canonical segmented handoff. It retains the segment-copy metadata while reading
one process-owned global payload snapshot, performs one checksum scan, and
feeds that global view directly to layer loading. The forced
`legacy_segment_remerge` control materializes segment views and merges them
again, performs two checksum scans, and accounts for reassembly copies equal to
twice the payload size.

Six isolated L4 jobs compare only the segmented-load strategy: unprofiled 8k
and 16k pairs plus a stage-profiled 8k pair. Each job contains six cold requests
over the same two examples and 11 independently generated document segments;
all 36 telemetry rows join their measurements one-to-one, report successful
page-cache eviction, and disable the payload cache. Within each pair, the
evidence proves that the complete effective suite plus the complete manifest
decoding and execution settings match. The `--representative-canary` and
`--representative-workload-profile` submission flags on the direct jobs select
validation policy only; the emitted effective suite and manifest are unchanged
relative to their generic legacy controls.

Artifact generation identity, geometry, topology, and byte counts also match
within each pair. The jobs regenerated their payloads separately, however, and
did not retain a cross-job payload-content checksum. Literal byte-for-byte
payload equality is therefore not independently verified.

| Input / output | Strategy | Profiled | P50 TTFT (s) | P50 provider load (ms) | P50 materialize (ms) | P50 merge (ms) | P50 layer load (ms) | Reassembly / request (MiB) |
| ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 / 64 | direct global snapshot | no | 2.909791 | 2726.820 | 2555.694 | 0.052 | 170.438 | 0.000 |
| 8192 / 64 | legacy segment remerge | no | 7.315308 | 7041.608 | 5000.655 | 1865.408 | 170.491 | 2292.750 |
| 16384 / 256 | direct global snapshot | no | 5.846156 | 5522.386 | 5149.911 | 0.052 | 370.769 | 0.000 |
| 16384 / 256 | legacy segment remerge | no | 15.264031 | 14762.074 | 10448.391 | 3909.231 | 369.262 | 4596.750 |
| 8192 / 64 | direct global snapshot | yes | 2.901357 | 2718.322 | 2544.796 | 0.056 | 173.492 | 0.000 |
| 8192 / 64 | legacy segment remerge | yes | 7.581663 | 7309.192 | 5281.612 | 1861.763 | 173.468 | 2292.750 |

| Matched pair | TTFT speedup | TTFT reduction | Provider-load speedup | Provider-load reduction |
| --- | ---: | ---: | ---: | ---: |
| 8k, unprofiled | 2.5140x | 60.22% | 2.5824x | 61.28% |
| 16k, unprofiled | 2.6110x | 61.70% | 2.6731x | 62.59% |
| 8k, stage-profiled diagnostic | 2.6131x | 61.73% | 2.6889x | 62.81% |

The unchanged layer-load medians in each pair show that the measured savings
come from CPU payload materialization and eliminated reassembly, not GPU layer
copy. At 8k, direct materialization is 93.7% of provider load, so it remains the
dominant optimization target. That timer combines cold disk first-touch,
creation of the owned snapshot, and checksum work; it is not a pure
disk-throughput measurement.

No prefetch event occurred in any of these six jobs. Thus this ablation proves
the direct-load optimization, not overlap between disk loading and model
prefill. The provider has an opt-in concurrent-prefetch path, but its behavior
and benefit require a separately matched ablation before making a performance
claim. Stage profiling synchronizes CUDA and can perturb end-to-end timing;
the profiled pair is diagnostic only. Its `h2d` and `scatter` values are
subcomponents of `layer_load`, not additional latency.

## SGLang Native-Handoff Smoke

The SGLang record validates the same Vanilla-v2 pre-RoPE position contract on
the native hierarchical-cache handoff. Its closed handoff-generation projection
binds `vanilla_prefill` v2 to the forced pre-RoPE generator factory (version
`5.3.0`), per-document topology, safe content/topology digests, and canonical
raw-sidecar SHA-256
`49cf15b2d53f55a9f48594c120dc1cafe9d905c407a51116c8b54d5606eb405a`.
No sidecar path, prompt, request ID, or raw text is retained. It is an execution
smoke, not a matching 4k-to-32-token benchmark: although the profile reserves
4k input and 32 output tokens, every request used 205 prompt tokens and
completed with 7 tokens. One synthetic NIAH example was repeated twice per arm.

| Arm | Requests | Actual prompt / completion | P50 TTFT (s) | P50 TTC (s) | Answer-found / exact match |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline prefill | 2 | 205 / 7 | 2.911591 | 3.167029 | 1.00 / 1.00 |
| document KV cache | 2 | 205 / 7 | 4.570751 | 4.826868 | 1.00 / 1.00 |

The cache arm is slower in this smoke: baseline-to-cache ratios are 0.6370 for
TTFT and 0.6561 for TTC. These four requests are not a performance claim. Both
cache requests report 176 cached tokens and 29 newly prefetched prompt tokens,
which confirms that request metadata reached the SGLang native cache path.

## Interpretation Limits

- The vLLM evidence covers only two HotpotQA examples; the SGLang evidence
  covers one synthetic NIAH example. Repeats are not independent samples.
- The canaries use BF16 throughout, request parallelism 1, and isolated arms.
  They do not match the main Q4-weight/Q8-document-KV, parallelism-4 protocol.
- The direct-load ablation varies a provider implementation strategy, not a
  cache method. The optimized canonical segmented loader can serve compatible
  methods through the vLLM provider, but only Vanilla v2 is evidenced here.
- No publication-grade resource-use or serving-concurrency claim is made.
- The L4 node has two 450 GB local NVMe disks; the A10G node has one 900 GB
  local NVMe disk. Their comparison is a compatibility observation, not a
  GPU-only hardware ablation.
- Publication still requires the exact main protocol, complete matched dataset
  coverage, adequate independent samples, predeclared statistics, resource
  telemetry, and a passing publication gate for the release wheel.

These files support reproducibility, regression diagnosis, and runtime
integration checks. They must not be cited as released Cachet quality,
latency, throughput, storage-bandwidth, or hardware-efficiency claims.
