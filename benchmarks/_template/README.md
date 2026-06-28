# Benchmark Report Template

Use this template for new public benchmark result folders under
`benchmarks/appendix/`. Folder names should be stable and descriptive, not date
or run-id based.

Do not infer or estimate missing values. Leave numeric cells blank when a
metric has not been measured under the stated configuration. Use `N/A` only
when a metric cannot apply. Blank cells and `N/A` are not zeros.

## Shared Table Configuration

Use this block for every main table in the report unless a specific table
caption documents an intentional override. Context length belongs in latency
tables by default; include context in score tables only when the scoring
protocol intentionally pads/truncates every dataset sample to those context
lengths.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507`, served as `qwen3:4b-instruct` unless this report explicitly varies the model |
| Model weights | 4-bit bitsandbytes unless this report explicitly varies weight precision |
| Serving engine | vLLM, SGLang, storage reader, native probe, etc. |
| Hardware | e.g. AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | e.g. 8 requests in flight, or `N/A` |
| Output length for TTC | e.g. forced 256-token decode, or `N/A` |
| Repeats | e.g. 512 repeats per prepared input, or `N/A` |
| Input context length | e.g. 8k, 16k, 32k, or measured prompt-token range |
| Method | Baseline, vanilla KV, KV Packet, etc. |
| Document KV precision | bf16, Q8 / `fp8_e5m2`, packed Q4, or `N/A` |
| Runtime KV dtype | e.g. `fp8_e5m2`, `bfloat16`, or `N/A` |
| Storage tier / cache residency | RAM, disk, Unity Catalog, hybrid, or `N/A` |
| TTFT measurement boundary | Cold disk-to-GPU hydrate, warm prewarmed prefix cache, RAM-resident hydrate, or `N/A` |
| Prefix-cache policy | Per-request `cache_salt`, static `cache_salt`, prefix caching disabled, or `N/A` |
| Dataset / task scope | Dataset names and example count |
| Quality metric | Full-dataset task score, answer-found containment, strict exact match, or `N/A` |
| Evidence file | Link to sanitized committed JSON |

## Latency And Resource Table

Use request-level percentiles for latency. If decode throughput is reported,
state whether it is end-to-end output throughput or decode-only throughput.
The preferred decode-only metric is
`completion_tokens / (time_to_completion_seconds - ttft_seconds)`.
Place the detailed caption below the table. The caption should define each
method label, state the request concurrency used during measurement, give the
successful request count behind percentiles, and explain whether P95 is
publication-grade. P95 rows intended for publication should use at least 512
repeats per prepared input at the stated concurrency, and the caption should
state the resulting successful request-level measurement count. The caption must
also say whether TTFT includes loading external document KV from storage into
GPU memory, or whether the measured requests used already-warm/prewarmed
prefix-cache blocks.

Latency values are seconds.

| Method | Input context | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 tok/s | Max Serving Concurrency | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Example&nbsp;method | 16k |  |  |  |  |  |  |  |

Use the same columns even when a result only covers a subset. If the result is
not a serving-latency benchmark, mark latency cells `N/A` and explain the scope
in `Limitations`. State whether Max Serving Concurrency is a direct server-log
value or derived from GPU KV-cache tokens divided by a nominal context length.
Do not use accounted GPU memory as a synonym for full sampled peak GPU process
memory. If an ablation varies document KV precision, add that as an
ablation-specific column in the ablation table rather than the main latency
table.

## Benchmark Dataset Score Table

Describe the dataset scope before the table. For main benchmark score tables,
evaluate all selected samples of each dataset and leave score cells blank until
those full-dataset runs are complete. For prepared smoke suites, use a
separate appendix table, state the number of unique examples per dataset and
repeats per example, and do not label answer-found containment as official
dataset accuracy.

| Method | Biography score | HotpotQA score | MusiQue score | NIAH score | LongBench v2 score | RULER score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Example&nbsp;method |  |  |  |  |  |  |

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
- `prewarm-cache-prefix.json`, only for warm/prewarmed-prefix measurements

Do not include Databricks tokens, raw Jobs API responses, package wheels,
cluster logs, generated payload blobs, prompt text, or local scratch
directories.
