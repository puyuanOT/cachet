# Benchmark Report Template

Use this template for new public benchmark result folders under
`benchmarks/appendix/`. Folder names should be stable and descriptive, not date
or run-id based.

Do not infer or estimate missing values. Leave numeric cells blank when a
metric has not been measured under the stated configuration. Use `N/A` only
when a metric cannot apply. Blank cells and `N/A` are not zeros.

## Table Configuration

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507`, served as `qwen3:4b-instruct` unless this report explicitly varies the model |
| Model weights | 4-bit bitsandbytes unless this report explicitly varies weight precision |
| Serving engine | vLLM, SGLang, storage reader, native probe, etc. |
| Hardware | e.g. AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | e.g. 8 requests in flight, or `N/A` |
| Output length for TTC | e.g. forced 256-token decode, or `N/A` |
| Input context length | e.g. 8k, 16k, 32k, or measured prompt-token range |
| Method | Baseline, Cachet + vanilla KV, Cachet + KV Packet, etc. |
| Document KV precision | bf16, Q8 / `fp8_e5m2`, packed Q4, or `N/A` |
| Runtime KV dtype | e.g. `fp8_e5m2`, `bfloat16`, or `N/A` |
| Storage tier / cache residency | RAM, disk, Unity Catalog, hybrid, or `N/A` |
| Dataset / task scope | Dataset names and example count |
| Quality metric | Answer-found containment, strict exact match, task score, or `N/A` |
| Evidence file | Link to sanitized committed JSON |

## Latency And Resource Table

Use request-level percentiles for latency. If decode throughput is reported,
state whether it is end-to-end output throughput or decode-only throughput.
The preferred decode-only metric is
`completion_tokens / (time_to_completion_seconds - ttft_seconds)`.

| Method | Input context | Document KV precision | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | P50 decode tok/s | vLLM KV capacity | Accounted GPU memory | Peak GPU process memory |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Example method | 16k | Q8 (`fp8_e5m2`) |  |  |  |  |  |  |  |  |

Use the same columns even when a result only covers a subset. If the result is
not a serving-latency benchmark, mark latency cells `N/A` and explain the scope
in `Limitations`. State whether vLLM KV capacity is a direct server-log value
or derived from GPU KV-cache tokens divided by a nominal context length. Do not
use accounted GPU memory as a synonym for full sampled peak GPU process memory.

## Prepared Dataset Quality Table

Describe the dataset scope before the table. For prepared smoke suites, state
the number of unique examples per dataset and repeats per example. Do not label
answer-found containment as official dataset accuracy.

| Method | Input context | Prepared examples per dataset | Repeats per example | Metric | Biography | HotpotQA | MusiQue | NIAH |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Example method | 16k |  |  | Answer-found / strict EM |  |  |  |  |

## Resource Utilization

| Experiment row | Storage tier | Peak GPU process memory | GPU utilization | Peak CPU RSS / host RAM | Disk read throughput | Network / Unity Catalog read throughput | KV cache footprint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Example method | Disk |  |  |  |  | N/A |  |

Do not use storage throughput as a synonym for memory consumption. Report disk
or Unity Catalog throughput only when the evidence directly measures those
readers. If only vLLM component telemetry is available, label it as accounted
GPU memory rather than peak GPU process memory.

## Limitations

| Limitation | Current state |
| --- | --- |
| Primary-table comparability | State whether this result matches the current Q4-weight + Q8-document-KV protocol |
| Model coverage | List covered models or say `not yet measured` |
| Method coverage | List covered methods; mark planned methods such as KV Packet as `not implemented yet` |
| Context coverage | List covered context lengths or prompt-token ranges |
| Precision coverage | List covered document KV precisions; explain any pending packed-Q4 support |
| Quality coverage | State whether quality rows are smoke checks, official dataset scores, or another metric |
| Resource metrics | Say which memory/utilization/cache-footprint fields are missing |

## Provenance

List sanitized records committed beside this README, such as:

- `summary.json`
- `v1-benchmark.json`
- `metadata.json`
- `document-kv-connector-telemetry.jsonl`
- `prepared-handoff-generation.json`
- `prewarm-cache-prefix.json`

Do not include Databricks tokens, raw Jobs API responses, package wheels,
cluster logs, generated payload blobs, prompt text, or local scratch
directories.
