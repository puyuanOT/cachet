# Representative BF16 Qwen3-4B Canaries

This folder contains sanitized, reproducible engineering canaries for the
generalized Cachet benchmark and native serving paths. They are deliberately
**non-publication-qualified** and do not populate or revise any value in the
[main Q4-weight/Q8-document-KV tables](../../README.md). These runs used BF16
model weights and BF16 runtime/document KV with no quantization, so their
latencies are not comparable to the main-table protocol.

## Bound Identity

| Field | Value |
| --- | --- |
| Source revision | `6e0f501a52c6b19f66d36e53a3fe6035b4b36ea2` |
| Cachet wheel SHA-256 | `d820c01c5bee4d3bcb1e4338e4081c1ea9b4b59c8cb725588d7b973c07fe6f47` |
| Model and tokenizer | `Qwen/Qwen3-4B-Instruct-2507` |
| Model and tokenizer revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Model weights / runtime KV / document KV | BF16 / BF16 / BF16; no quantization |
| vLLM | `0.23.0` |
| SGLang | `0.5.10.post1` |

The canonical records retain the exact runtime package pins, hardware
fingerprints, logical-sample digests, physical-transform identities, decoding
settings, and sanitized request correlations.

## Evidence Files

| File | Scope | Canonical file SHA-256 |
| --- | --- | --- |
| [`g6-vllm-8k-64-three-arm-canary.json`](g6-vllm-8k-64-three-arm-canary.json) | L4, 8k input target, 64 output tokens | `2fd6d83fa7ef587af70559e6f0d615ecdc7710d93c7c7739e743f1df56bacae5` |
| [`g6-vllm-16k-256-three-arm-canary.json`](g6-vllm-16k-256-three-arm-canary.json) | L4, 16k input target, 256 output tokens | `8396ad63b8d0bf606f355ef49fcb370b3837a124fe401a214dad05d4b8926709` |
| [`g5-vllm-8k-64-three-arm-canary.json`](g5-vllm-8k-64-three-arm-canary.json) | A10G compatibility canary, 8k input target, 64 output tokens | `722ee6ae0bcc6cd7a092ef9039e7f1ad4db4433b649849bfef949fc1a4de68e8` |
| [`g6-sglang-4k-32-paired-smoke-evidence.json`](g6-sglang-4k-32-paired-smoke-evidence.json) | L4 native-handoff execution smoke | `3266485e74047d638577d3cd1cfa85c4080650eaf286247d22a56a03810238fe` |

## vLLM Three-Arm Canaries

Each vLLM arm ran in its own single-node job with request parallelism 1, zero
warmups, and the same `2 examples × 3 repeats`. Thus each arm has six requests,
but the repeats are repeated measurements of two examples, not six independent
samples. The arms are:

- `baseline_prefill`: standard no-cache full-prompt recomputation.
- `full_prefix_prefill`: an exact full-prefix control materialized as one
  reusable segment.
- `vanilla_prefill`: independently materialized per-document segments.

The distinction is recorded in each manifest's physical transform and method
configuration. It is not an undocumented request-path optimization.

Latency values below are seconds. TTFT and TTC ratios are the record's
baseline-to-arm comparisons, computed from arm P50 values; **a ratio below 1 means
the cache arm was slower than baseline**. The table shows P50 latency and mean
HotpotQA F1 for compactness; the JSON records retain every measurement, P95,
paired statistic, and full-precision value.

| Hardware | Arm | Requests | P50 TTFT | P50 TTC | Mean F1 | TTFT ratio | TTC ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `g6.8xlarge` / L4 | baseline | 6 | 1.525 | 3.890 | 0.05625 | 1.000 | 1.000 |
| `g6.8xlarge` / L4 | full-prefix control | 6 | 2.429 | 4.791 | 0.05530 | 0.628 | 0.812 |
| `g6.8xlarge` / L4 | vanilla per-document | 6 | 7.250 | 9.612 | 0.03333 | 0.210 | 0.405 |
| `g5.8xlarge` / A10G | baseline | 6 | 1.410 | 2.762 | 0.05625 | 1.000 | 1.000 |
| `g5.8xlarge` / A10G | full-prefix control | 6 | 2.489 | 3.840 | 0.05625 | 0.567 | 0.719 |
| `g5.8xlarge` / A10G | vanilla per-document | 6 | 6.668 | 8.021 | 0.03704 | 0.212 | 0.344 |

Both 8k trios completed with zero request errors. For each trio, the sanitized
cache measurements and cold-read attestations join one-to-one: 12 of 12 cache
requests are cold-attested, with successful page-cache eviction, no payload
cache hit, and one successful load. That validates the recorded cold-read
correlation; it does not turn this small canary into publication evidence.
The embedded canary-policy gate reports zero cold-attested requests because
that preset does not consume cold evidence; the committed validator separately
replays the publication correlation contract and verifies the 12-of-12 join.

Both 8k canary gates fail with this issue, retained verbatim from the records:

> hotpotqa:document_kv_cache:vanilla_prefill paired 'f1' lower bound -0.0625 exceeds allowed regression 0.02

The g6 run also records `post-success vLLM forced-shutdown EngineDeadError`.
It occurred during forced server teardown after successful measurements and is
reported as an operational caveat, not hidden or counted as a request failure.
The g5 run does not carry that teardown caveat.

## g6 16k Result

The 16k trio also completed with zero request errors and exact 16,384-prompt /
256-completion-token server accounting for all 18 requests. Its 12 cache
measurements join one-to-one with 12 mechanically cold attestations. Each
full-prefix request loaded 16,336 block-aligned tokens from one segment; each
vanilla request loaded the same token count from 11 independently generated
document segments.

| Arm | Requests | P50 TTFT | P50 TTC | Mean F1 | TTFT ratio | TTC ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 6 | 3.795 | 14.624 | 0.01474 | 1.000 | 1.000 |
| full-prefix control | 6 | 4.578 | 15.405 | 0.01392 | 0.829 | 0.949 |
| vanilla per-document | 6 | 14.698 | 25.528 | 0.00926 | 0.258 | 0.573 |

The 16k canary gate passes with no issues. Both cache arms are nevertheless
slower than baseline in this BF16, request-parallelism-1 setting, and the small
two-example canary remains non-publication-qualified. No teardown caveat was
observed for this trio.

## SGLang Native-Handoff Smoke

The SGLang record is an execution smoke, not a matching 4k-to-32-token latency
benchmark. Although the workload profile reserves a 4k input target and a
32-token output budget, server usage recorded **205 prompt tokens and 7
completion tokens** for every measured request. It contains one synthetic NIAH
example repeated twice per arm; those two repeats are not independent samples.

| Arm | Requests | Actual prompt / completion tokens | P50 TTFT | P50 TTC | Answer-found / exact match |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline prefill | 2 | 205 / 7 | 4.319 | 4.572 | 1.00 / 1.00 |
| document KV cache | 2 | 205 / 7 | 4.077 | 4.331 | 1.00 / 1.00 |

The corresponding TTFT and TTC ratios are 1.059 and 1.056. Those values only
describe these four smoke requests; they are not a performance claim. The
native validation observed 176 cached tokens and 29 newly prefetched prompt
tokens on the cache request, confirming that request metadata reached the
SGLang hierarchical-cache path.

## Interpretation Limits

- `v1_evidence.ok=false` is expected for the vLLM subset canaries. A passing
  canary execution or a valid cold-attestation join is not a V1 publication
  gate.
- The vLLM evidence uses only two HotpotQA examples, and the SGLang evidence
  uses one synthetic NIAH example. Neither covers the required datasets or
  provides publication-scale independent samples and confidence bounds.
- No resource-use claim is made. These records do not establish a
  publication-grade peak GPU-memory, CPU-memory, disk-bandwidth, or serving
  concurrency comparison.
- The g6/L4 node has two 450 GB local NVMe disks, while the g5/A10G node has one
  900 GB local NVMe disk. Because disk topology and the NVMe-to-CPU-memory path
  differ, the g5/g6 comparison is a separate-setting compatibility observation,
  not a GPU-only hardware ablation.
- Publication still requires the main Q4-weight/Q8-document-KV setting,
  complete and matched dataset coverage, adequate independent sample counts,
  predeclared statistical analysis, resource telemetry, a passing publication
  gate, and refreshed sanitized evidence for the exact release wheel.

Accordingly, these files are useful for reproducibility, regression diagnosis,
and method/runtime integration checks, but must not be cited as released Cachet
quality, latency, throughput, or hardware-efficiency claims.
