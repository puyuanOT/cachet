# Benchmark Report Template

Use this template for new public benchmark result folders under
`benchmarks/appendix/`. Folder names should be stable and descriptive, not date
or run-id based.

Do not infer or estimate missing values. Use `N/A (reason)` when a metric is
unsupported, deliberately unrun, or cannot apply. Reserve blank numeric cells
for an active run that is expected to be filled before the report is finalized.
Neither blank cells nor `N/A` are zeros.

Label the report as **smoke**, **canary**, or **publication** according to
[`docs/evidence-policy.md`](../../docs/evidence-policy.md). Private DBFS paths
and run IDs may supplement provenance but never substitute for sanitized
committed measurements and a publication-gate result. If those records are
missing, label every populated comparison **provisional /
non-publication-qualified**.

## Shared Table Configuration

Use this block for every main table in the report unless a specific table
caption documents an intentional override. Context length belongs in latency
tables by default; include context in score tables only when the scoring
protocol intentionally pads/truncates every dataset sample to those context
lengths.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507`, served as `qwen3:4b-instruct` unless this report explicitly varies the model |
| Model weights | bitsandbytes 4-bit runtime weights; the model identifier is not a prequantized 4-bit checkpoint |
| Serving engine | vLLM 0.27.1 for the frozen campaign, or an explicitly declared ablation target |
| Hardware | e.g. AWS g6/L4, `g6.8xlarge` |
| Closed-loop request concurrency | 1, 2, or 4 with zero think time, or `N/A` |
| Output length for TTC | e.g. forced 256-token decode, or `N/A` |
| Distinct latency examples | 32 per dataset, or `N/A` |
| Repeats | 2 per example within each deployment block, or `N/A` |
| Deployment blocks | 5 matched fresh-cluster blocks, or `N/A` |
| Input context length | e.g. 8k, 16k, 32k, or measured prompt-token range |
| Method | Baseline, Vanilla KV, KV Packet, etc. |
| Method ID / version | Stable `MethodSpec.method_id` and artifact-semantics version |
| Artifact ID | SHA-256 `ArtifactIdentity.artifact_id`, or `N/A` for baseline |
| Document KV precision | bf16, Q8 / `fp8_e5m2`, packed Q4, or `N/A` |
| Runtime KV dtype | e.g. `fp8_e5m2`, `bfloat16`, or `N/A` |
| Storage tier / cache residency | RAM, disk, Unity Catalog, hybrid, or `N/A` |
| TTFT measurement boundary | Cold disk-to-GPU hydrate, warm prewarmed prefix cache, RAM-resident hydrate, or `N/A` |
| Prefix-cache policy | Per-request `cache_salt`, static `cache_salt`, prefix caching disabled, or `N/A` |
| Dataset / task scope | Dataset names and example count |
| Quality metric | Versioned governed full-dataset metric and answer-parser identity, or `N/A` |
| Evidence file | Link to sanitized committed JSON |
| Campaign report | Sealed `cachet.vllm_0271_publication_report.v1` record, or `N/A` outside that campaign |
| Publication gate | Passing `document_kv.benchmark_publication_gate.v1` record |

For method comparisons, fix the complete setting and vary only the method. For
ablations, fix the method and vary exactly one declared factor. Record the
common logical input plus every method-specific physical transformation and
physical token accounting. Report offline training/artifact-generation costs
and artifact footprint separately from online TTFT/TTC, and state whether
storage-to-CPU and CPU-to-GPU hydration are inside the measured boundary.

## Latency And Resource Table

Use request-level percentiles for latency. If decode throughput is reported,
state whether it is end-to-end output throughput or decode-only throughput.
The preferred decode-only metric is
`completion_tokens / (time_to_completion_seconds - ttft_seconds)`.
Place the detailed caption below the table. The caption should define each
method label, state the request concurrency used during measurement, give the
successful request count behind percentiles, and explain whether P95 is
publication-grade. The frozen campaign uses 32 distinct examples per dataset,
two repeats per example, and 256 successful requests per
method/context/concurrency cell in each of five matched deployment blocks. The
caption must also say whether TTFT includes loading external document KV from
storage into GPU memory, or whether the measured requests used already-warm
prefix-cache blocks.

Latency values are seconds.

| Method | Input context | Concurrency | P50 TTFT | P95 TTFT | P50 TTC (256 toks) | P95 TTC (256 toks) | P50 decode tok/s | Configured closed-loop concurrency | Peak GPU process memory | Peak host memory | Peak process-tree RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Example&nbsp;method | 16k | 4 |  |  |  |  |  |  |  |  |  |

Use the same columns even when a result only covers a subset. If the result is
not a serving-latency benchmark, mark latency cells `N/A` and explain the scope
in `Limitations`. State whether Max Serving Concurrency is a direct server-log
value or derived from GPU KV-cache tokens divided by a nominal context length.
Do not use accounted GPU memory as a synonym for full sampled peak GPU process
memory. If an ablation varies document KV precision, add that as an
ablation-specific column in the ablation table rather than the main latency
table.

## Paired Latency Estimands

Report every preregistered comparison as `reference latency / treatment
latency`, so a speedup above 1 means the treatment is faster. Include separate
TTFT and TTC geometric speedups and pointwise 95% paired hierarchical-bootstrap
intervals. The 0.27.1 campaign has 13 rows: nine Baseline/Vanilla
context-by-concurrency comparisons plus BF16/Q8, RAM/Disk, Unity Catalog/Disk,
and A10G/L4.

| Treatment vs reference | Setting | TTFT geometric speedup | TTFT 95% CI | TTC geometric speedup | TTC 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| Example treatment vs control | 16k, concurrency 4 |  |  |  |  |

## Benchmark Dataset Score Table

Describe the dataset scope before the table. For main benchmark score tables,
evaluate all selected samples of each dataset and mark score cells
`N/A (full dataset not run)` until those runs are complete. For prepared smoke
suites, use a
separate appendix table, state the number of unique examples per dataset and
repeats per example, and do not label answer-found containment as official
dataset accuracy. Publication score tables use six decimal places and must use
scientific notation rather than displaying zero when a nonzero value would
round to zero.

| Dataset | Governed metric | n | Baseline | Baseline parser-status counts | Vanilla KV | Vanilla parser-status counts | Vanilla − Baseline | Paired 95% CI |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Biography | Normalized-title exact match |  |  |  |  |  |  |  |
| HotpotQA | Answer exact match |  |  |  |  |  |  |  |
| HotpotQA | Answer F1 |  |  |  |  |  |  |  |
| MusiQue | Official answer exact match, alias-max |  |  |  |  |  |  |  |
| MusiQue | Official answer F1, alias-max |  |  |  |  |  |  |  |
| NIAH | Exact-value overall accuracy |  |  |  |  |  |  |  |

Publish the NIAH 8k/16k/32k by 10%/50%/90% needle-position grid separately;
an overall score does not replace any of its nine governed cells. Each grid row
must report `n`, both arm accuracies, the paired delta, and its paired 95% CI.

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
| Primary-table comparability | State whether this result matches the frozen vLLM 0.27.1, bitsandbytes 4-bit-runtime-weight, Q8-document-KV protocol |
| Model coverage | List covered models or say `not yet measured` |
| Method coverage | List covered methods and mark genuinely unimplemented methods explicitly |
| Context coverage | List covered context lengths or prompt-token ranges |
| Precision coverage | List covered document KV precisions; treat pending packed-Q4 support as compression rather than a KV Packet method label |
| Quality coverage | State whether quality rows are smoke checks, official dataset scores, or another metric |
| Resource metrics | Say which memory/utilization/cache-footprint fields are missing |

## Provenance

List sanitized records committed beside this README, such as:

- `summary.json`
- `v1-benchmark.json`
- `metadata.json`
- `document-kv-connector-telemetry.jsonl`
- `cache-state-attestations.jsonl`
- `campaign-report.json`
- `benchmark-publication-gate.json`
- `prepared-handoff-generation.json`
- `prewarm-cache-prefix.json`, only for warm/prewarmed-prefix measurements

For the vLLM 0.27.1 campaign, `campaign-report.json` and
`benchmark-publication-gate.json` are an exact pair: the gate's
`benchmark_payload_digest` must equal the report's `closed_record_sha256`.
Neither file qualifies evidence by itself. Publication writes the gate first
and the report last as the pair's commit file; only an exact immutable
gate-only interruption may be resumed. The authority-replaying loader accepts
the sealed files directly or secure-umask Git checkout copies (`0600`, `0640`,
or `0644`, including read-only variants), but never uses checkout mode as a
substitute for canonical-byte and source replay.

For the frozen campaign, do not transcribe report values into Markdown by
hand. Generate the named table regions from the exact report/gate pair and
require byte-for-byte renderer validation before committing the report folder.
Generate the child `README.md` from the same pair; it must name both JSON files
and embed the validated report digest exactly.
Pending and finalized table states are mutually exclusive; partial promotion
or a mixed pending/populated table is invalid.

Do not include Databricks tokens, raw Jobs API responses, package wheels,
cluster logs, generated payload blobs, prompt text, or local scratch
directories.
