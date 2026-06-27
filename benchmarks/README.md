# Cachet Benchmarks

This directory is the public benchmark appendix for Cachet. It follows a
research-paper structure: one primary comparison table first, followed by
focused ablation tables. No values are inferred or estimated. A blank numeric
cell means not measured yet; not zero.

The filled baseline latency rows are backed by sanitized evidence in
[`appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/`](appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/).
That folder also contains full-logical-prompt Cachet handoff-path evidence,
which is useful appendix evidence but does not satisfy the primary Cachet
latency definition below. Historical evidence in
[`appendix/existing-results/`](appendix/existing-results/) does not match the
primary-table configuration below and should be treated as prior evidence only.

## Main Table Configuration

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct, `qwen3:4b-instruct` |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |
| Input context lengths | 8k, 16k, 32k tokens |
| KV cache residency for Cachet methods | Local disk, not Unity Catalog |
| Datasets / score columns | Biography, HotpotQA, MusiQue, NIAH |
| Score metric | Pending full-dataset evaluation; synthetic prepared-input sanity runs report `answer_found_rate` only |
| Baseline | No precomputed KV cache |
| Cachet methods | Vanilla external KV; KV Packet planned but not implemented yet |
| Cachet latency criterion | Engine request should bind cached KV out of band and compute only uncached prompt suffix plus generated tokens |
| Baseline evidence | [`appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/summary.json`](appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/summary.json) |
| Runtime-prompt canary | [`appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/failure_summary.json`](appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/failure_summary.json) |

## Main Performance Table

Latency values are seconds. Baseline percentiles are computed over 32
successful request-level measurements for each context length: four synthetic
prepared inputs times eight repeats. Cachet + vanilla KV latency cells are
blank because the current vLLM native provider requires the full logical prompt
for external-KV loads, while this primary table is intended to measure
suffix-only inference over precomputed KV. A Databricks canary that enabled the
runtime-prompt path failed with an explicit vLLM provider guard, so the older
full-logical-prompt Cachet handoff latencies stay in appendix evidence rather
than in the main comparison table. `Cachet + KV Packet` is listed for protocol
completeness and is not implemented yet.

| Method | Input context | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | 8k | 4.22 | 9.93 | 20.53 | 28.15 |  |  |  |  |
| Baseline, no precomputed KV | 16k | 25.40 | 33.43 | 41.32 | 58.02 |  |  |  |  |
| Baseline, no precomputed KV | 32k | 97.98 | 98.61 | 117.05 | 117.07 |  |  |  |  |
| Cachet + vanilla KV | 8k |  |  |  |  |  |  |  |  |
| Cachet + vanilla KV | 16k |  |  |  |  |  |  |  |  |
| Cachet + vanilla KV | 32k |  |  |  |  |  |  |  |  |
| Cachet + KV Packet | 8k |  |  |  |  |  |  |  |  |
| Cachet + KV Packet | 16k |  |  |  |  |  |  |  |  |
| Cachet + KV Packet | 32k |  |  |  |  |  |  |  |  |

Dataset score columns are intentionally blank because this run used one
synthetic prepared example per dataset/context, not full Biography, HotpotQA,
MusiQue, or NIAH evaluation sets. The committed evidence reports
`answer_found_rate=1.00` for those synthetic sanity examples, but that should
not be interpreted as dataset accuracy.

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

| Storage tier | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAM |  |  |  |  |  |  |  |  |
| Disk |  |  |  |  |  |  |  |  |
| Unity Catalog |  |  |  |  |  |  |  |  |
| Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  |  |  |

## Hardware Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Serving engine | vLLM |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Storage tier | Disk |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |

| Hardware | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AWS g5/A10G, `g5.8xlarge` |  |  |  |  |  |  |  |  |
| AWS g6/L4, `g6.8xlarge` |  |  |  |  |  |  |  |  |

## Serving Platform Ablation

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Method | Cachet + vanilla KV |
| Input context | 16k tokens |
| Storage tier | Disk |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Emit 256 tokens |

| Serving platform | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM |  |  |  |  |  |  |  |  |
| SGLang |  |  |  |  |  |  |  |  |

## Resource Utilization

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Default engine / hardware | vLLM on AWS g5/A10G, `g5.8xlarge` |
| Default context / output | 16k input context, 256 emitted tokens |
| Request parallelism | 8 requests in flight |

| Experiment row | Storage tier | Peak GPU memory | GPU utilization | Peak CPU RSS / host RAM | Disk read throughput | Network / Unity Catalog read throughput | KV cache footprint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | N/A |  |  |  |  |  |  |
| Cachet + vanilla KV | Disk |  |  |  |  |  |  |
| Cachet + vanilla KV | RAM |  |  |  |  |  |  |
| Cachet + vanilla KV | Unity Catalog |  |  |  |  |  |  |
| Cachet + vanilla KV | Hybrid RAM / disk / Unity Catalog |  |  |  |  |  |  |
| Cachet + KV Packet | Disk |  |  |  |  |  |  |

The `Cachet + KV Packet` resource-utilization row is reserved for a future
implementation.

## Primary Table Evidence

| Evidence folder | Evidence summary | Notes |
| --- | --- | --- |
| [`primary-table-vllm-qwen3-4b-g5-a10g-disk-cache`](appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/) | Baseline latency evidence plus full-logical-prompt vLLM Cachet handoff-path evidence at 8k, 16k, and 32k | Baseline rows fill the primary table; Cachet rows in this folder do not because the engine still received the full logical prompt |
| [`runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary`](appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/) | Failed 8k Cachet + vanilla KV canary with `--benchmark-cache-runtime-prompt` | vLLM rejected `document_kv.prompt_text_mode='runtime'`, so the primary Cachet latency rows remain unmeasured |

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
| [`appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/`](appendix/primary-table-vllm-qwen3-4b-g5-a10g-disk-cache/) | Sanitized baseline evidence and full-logical-prompt vLLM handoff-path evidence |
| [`appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/`](appendix/runtime-prompt-vllm-qwen3-4b-g5-a10g-disk-cache-canary/) | Failed runtime-prompt vLLM canary explaining why primary Cachet latency cells are blank |
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
