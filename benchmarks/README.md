# Cachet Benchmarks

This directory is the public benchmark appendix for Cachet. It follows a
research-paper structure: one primary comparison table first, followed by
focused ablation tables. No values are inferred or estimated.

The main table is backed by sanitized evidence in
[`appendix/primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/`](appendix/primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/).
Historical evidence in
[`appendix/existing-results/`](appendix/existing-results/) does not match the
primary-table configuration below and should be treated as prior evidence only.

## Main Table Configuration

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct, `qwen3:4b-instruct` |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Forced 256-token decode with `max_tokens=256` and `ignore_eos=true` |
| Input context lengths | 8k, 16k, 32k class prepared prompts; tokenizer counts are recorded in evidence |
| KV cache residency for Cachet methods | Local disk, not Unity Catalog |
| Datasets / quality columns | Biography, HotpotQA, MusiQue, NIAH |
| Quality metric | Prepared-suite exact match (EM) from natural-stop quality runs; not full-dataset accuracy |
| Baseline | No precomputed KV cache |
| Cachet methods | Vanilla external KV; KV Packet planned but not implemented yet |
| Cachet latency criterion | vLLM request attaches Cachet KV-transfer metadata, reports external prefix cache hits, and imports local-disk vanilla KV blocks for the cached prefix |
| Prompt mode for Cachet rows | `logical`; suffix-only runtime prompts are not supported by the current vLLM native provider |
| Main evidence | [`appendix/primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/summary.json`](appendix/primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/summary.json) |
| Runtime-prompt canary | [`appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/failure_summary.json`](appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/failure_summary.json) |

## Main Performance Table

Latency values are seconds. TTFT and TTC are computed from forced-256 latency
runs where every successful request emitted exactly 256 completion tokens.
Percentiles are computed over 32 successful request-level measurements per
method/context pair: four prepared synthetic inputs times eight repeats. EM is
exact-match rate from natural-stop quality runs over the same prepared
row/context. `N/A` means the method is not implemented or the ablation has not
been measured under the fixed configuration; it is not a zero.

| Method | Input context | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography EM | HotpotQA EM | MusiQue EM | NIAH EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | 8k | 2.73 | 9.89 | 21.38 | 28.51 | 1.00 | 0.00 | 0.00 | 0.00 |
| Baseline, no precomputed KV | 16k | 29.43 | 34.12 | 49.13 | 53.83 | 0.00 | 0.00 | 0.00 | 0.00 |
| Baseline, no precomputed KV | 32k | 96.13 | 97.31 | 114.61 | 114.93 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cachet + vanilla KV | 8k | 7.91 | 8.17 | 17.74 | 17.98 | 1.00 | 0.00 | 0.00 | 0.00 |
| Cachet + vanilla KV | 16k | 27.54 | 28.03 | 37.31 | 37.79 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cachet + vanilla KV | 32k | 57.71 | 62.08 | 71.28 | 71.67 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cachet + KV Packet | 8k | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Cachet + KV Packet | 16k | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Cachet + KV Packet | 32k | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

The EM columns are prepared-suite quality checks, not full-dataset benchmark
accuracy. Raw evidence also records `answer_found_rate`; it is intentionally
not used as the main quality metric because it can make permissive generations
look artificially perfect.

The Cachet rows are current vLLM external-prefix measurements. They load raw
vanilla KV from local disk and skip cached-token prefill, but vLLM still
allocates GPU KV cache for the full logical context. On `g5.8xlarge`, the 32k
run reports 82,960 GPU KV-cache tokens available and 2.53x maximum concurrency
for 32,768-token requests, so an 8-way latency test queues in waves.
Baseline rows use the same connector-enabled vLLM server for parity, but
baseline requests do not attach Cachet KV-transfer parameters; server evidence
reports 0% external-prefix cache hits for those rows.

## Storage Tier Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Forced 256-token decode |

| Storage tier | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography EM | HotpotQA EM | MusiQue EM | NIAH EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAM | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Disk | 27.54 | 28.03 | 37.31 | 37.79 | 0.00 | 0.00 | 0.00 | 0.00 |
| Unity Catalog | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Hybrid RAM / disk / Unity Catalog | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Hardware Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Serving engine | vLLM |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Storage tier | Disk |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Forced 256-token decode |

| Hardware | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography EM | HotpotQA EM | MusiQue EM | NIAH EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AWS g5/A10G, `g5.8xlarge` | 27.54 | 28.03 | 37.31 | 37.79 | 0.00 | 0.00 | 0.00 | 0.00 |
| AWS g6/L4, `g6.8xlarge` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Serving Platform Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Storage tier | Disk |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Forced 256-token decode |

| Serving platform | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography EM | HotpotQA EM | MusiQue EM | NIAH EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM | 27.54 | 28.03 | 37.31 | 37.79 | 0.00 | 0.00 | 0.00 | 0.00 |
| SGLang | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Resource Utilization

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Default engine / hardware | vLLM on AWS g5/A10G, `g5.8xlarge` |
| Default context / output | 16k input context, 256 emitted tokens |
| Request parallelism | 8 requests in flight |

| Experiment row | Storage tier | GPU KV-cache pool | GPU KV-cache usage | Peak CPU RSS / host RAM | Disk payload read | Network / Unity Catalog read | KV payload footprint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | N/A | 11.39 GiB | 81.6% max | N/A | N/A | N/A | N/A |
| Cachet + vanilla KV | Disk | 11.39 GiB | 81.7% max | N/A | 1.35s p50 for 2.29 GiB | N/A | 2.29 GiB p50 |
| Cachet + vanilla KV | RAM | N/A | N/A | N/A | N/A | N/A | N/A |
| Cachet + vanilla KV | Unity Catalog | N/A | N/A | N/A | N/A | N/A | N/A |
| Cachet + vanilla KV | Hybrid RAM / disk / Unity Catalog | N/A | N/A | N/A | N/A | N/A | N/A |
| Cachet + KV Packet | Disk | N/A | N/A | N/A | N/A | N/A | N/A |

The `Cachet + KV Packet` resource-utilization row is reserved for a future
implementation.

## Primary Table Evidence

| Evidence folder | Evidence summary | Notes |
| --- | --- | --- |
| [`primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry`](appendix/primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/) | Current main-table evidence for baseline and Cachet + vanilla KV at 8k, 16k, and 32k | Separates forced-256 latency from natural-stop quality and includes Cachet provider-load telemetry |
| [`runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary`](appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/) | Failed 8k Cachet + vanilla KV canary with `--benchmark-cache-runtime-prompt` | Documents why the current vLLM provider uses logical prompt text for external-KV loads |
| [`primary-table-v2-vllm-qwen3-4b-g5-a10g-disk-cache`](appendix/primary-table-v2-vllm-qwen3-4b-g5-a10g-disk-cache/) | Superseded primary-table evidence | Kept because its numbers were mostly comparable, but the 8k row did not force every request to emit 256 tokens and it lacked provider telemetry |
| [`primary-table-vllm-qwen3-4b-g5-a10g-disk-cache`](appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/) | Superseded baseline and full-logical-prompt Cachet evidence from the earlier prepared suite | Kept as historical evidence from an earlier prepared-suite pass |

## Appendix Evidence

These additional committed results remain useful, but they do not satisfy the
primary-table configuration because they used different prompt-token ranges,
repeat counts, output lengths, cache/storage assumptions, hardware, or serving
paths.

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
| [`appendix/primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/`](appendix/primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/) | Current sanitized main-table evidence |
| [`appendix/primary-table-v2-vllm-qwen3-4b-g5-a10g-disk-cache/`](appendix/primary-table-v2-vllm-qwen3-4b-g5-a10g-disk-cache/) | Superseded sanitized main-table evidence without forced-256 validation on every row |
| [`appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/`](appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/) | Superseded sanitized main-table evidence from an earlier prepared suite |
| [`appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/`](appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/) | Failed runtime-prompt vLLM canary for suffix-only prompt text |
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
