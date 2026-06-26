# Benchmark Report Template

Use this template for new public benchmark result folders under
`benchmarks/appendix/existing-results/` or another benchmark appendix subtree.
Folder names should be stable and descriptive, not date or run-id based.

Do not invent missing numbers. If a metric is absent from committed evidence,
leave the numeric cell blank and explain the status in the `Status / evidence`
column. A blank numeric cell means not measured yet; not zero.

## Table Configuration

| Field | Value |
| --- | --- |
| Model | e.g. Qwen3-4B-Instruct, `qwen3:4b-instruct` |
| Serving engine | vLLM, SGLang, storage reader, native probe, etc. |
| Hardware | e.g. AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | e.g. 8 requests in flight, or `n/a` |
| Output length for TTC | e.g. emit 256 tokens, or `n/a` |
| Input context length | e.g. 8k, 16k, 32k, or measured prompt-token range |
| Method | Baseline, Cachet + vanilla KV, Cachet + KV Packet, etc. |
| Cache location | RAM, disk, Unity Catalog, hybrid, or `n/a` |
| Dataset / task scope | Dataset names and example count |
| Evidence file | Link to sanitized committed JSON |

## Main Result Table

| Method | Input context | Cache location | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography score | HotpotQA score | MusiQue score | NIAH score | Status / evidence |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Example method | 16k | Disk |  |  |  |  |  |  |  |  | Not measured yet |

Use the same columns even when a result only covers a subset. If the result is
not a serving-latency benchmark, mark latency cells blank and explain the scope
in `Status / evidence`.

## Resource Utilization

| Method or ablation row | Cache location | Peak GPU memory | GPU utilization | CPU RSS / RAM | Disk read throughput | Network / UC read throughput | Cache footprint | Status / evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Example method | Disk |  |  |  |  |  |  | Not measured yet |

Do not use storage throughput as a synonym for memory consumption. Report disk
or Unity Catalog throughput only when the evidence directly measures those
readers.

## Limitations

| Limitation | Current state |
| --- | --- |
| Main-table comparability | State whether this result matches the fixed main-table configuration |
| Model coverage | List covered models or say `not benchmarked yet` |
| Method coverage | List covered methods; mark planned methods such as KV Packet as `not implemented yet` |
| Context coverage | List covered context lengths or prompt-token ranges |
| Resource metrics | Say which memory/utilization/cache-footprint fields are missing |

## Provenance

List sanitized records committed beside this README, such as:

- `v1_benchmark.json`
- `success_run.json`
- `storage_benchmark.json`
- `databricks_run_status.json`
- `release_evidence.json`
- `*_engine_probe.json`
- `*_connector_actions.json`

Do not include Databricks tokens, raw Jobs API responses, package wheels,
cluster logs, generated payload blobs, prompt text, or local scratch
directories.
