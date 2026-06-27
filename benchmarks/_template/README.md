# Benchmark Report Template

Use this template for new public benchmark result folders under
`benchmarks/appendix/existing-results/` or another benchmark appendix subtree.
Folder names should be stable and descriptive, not date or run-id based.

Do not infer or estimate missing values. If a metric is absent from committed
evidence, write `N/M` for not measured or `N/A` when the metric does not apply,
and document the limitation below. `N/M` and `N/A` are not zeros.

## Table Configuration

| Field | Value |
| --- | --- |
| Model | e.g. Qwen3-4B-Instruct, `qwen3:4b-instruct` |
| Serving engine | vLLM, SGLang, storage reader, native probe, etc. |
| Hardware | e.g. AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | e.g. 8 requests in flight, or `N/A` |
| Output length for TTC | e.g. emit 256 tokens, or `N/A` |
| Input context length | e.g. 8k, 16k, 32k, or measured prompt-token range |
| Method | Baseline, Cachet + vanilla KV, Cachet + KV Packet, etc. |
| Storage tier / cache residency | RAM, disk, Unity Catalog, hybrid, or `N/A` |
| Dataset / task scope | Dataset names and example count |
| Quality metric | Exact match (EM), answer-found, task score, or `N/A` |
| Evidence file | Link to sanitized committed JSON |

## Main Result Table

| Method | Input context | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 tokens) | P95 TTC (s, 256 tokens) | Biography EM | HotpotQA EM | MusiQue EM | NIAH EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Example method | 16k | N/M | N/M | N/M | N/M | N/M | N/M | N/M | N/M |

Use the same columns even when a result only covers a subset. If the result is
not a serving-latency benchmark, mark latency cells `N/A` and explain the scope
in `Limitations`.

## Resource Utilization

| Experiment row | Storage tier | Peak GPU memory | GPU utilization | Peak CPU RSS / host RAM | Disk read throughput | Network / Unity Catalog read throughput | KV cache footprint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Example method | Disk | N/M | N/M | N/M | N/M | N/A | N/M |

Do not use storage throughput as a synonym for memory consumption. Report disk
or Unity Catalog throughput only when the evidence directly measures those
readers.

## Limitations

| Limitation | Current state |
| --- | --- |
| Primary-table comparability | State whether this result matches the primary-table configuration |
| Model coverage | List covered models or say `not yet measured` |
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
