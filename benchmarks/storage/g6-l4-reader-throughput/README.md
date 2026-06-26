# Storage Reader Throughput On g6/L4

Reader throughput benchmark for Cachet memory, disk, and real Unity Catalog
Volume storage on the g6/L4 target.

## Experimental Setup

| Field | Value |
| --- | --- |
| Engine | Cachet storage readers |
| Model | n/a |
| Hardware | AWS g6/L4, `g6.8xlarge` |
| Method | Memory, disk, and Unity Catalog readers |
| Baseline arm | n/a |
| Cache arm | n/a |
| Dataset scope | 268,435,456-byte read workload per reader |
| Repeats / measurements | 256 reads; parallelism 8 |
| Evidence file | [`storage_benchmark.json`](storage_benchmark.json) |

## Reader Throughput And Latency

| Reader | Total bytes | Throughput | p50 latency | p95 latency | Reads | Parallelism | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Memory | 268,435,456 | 6531.4 MiB/s | 0.847 ms | 1.649 ms | 256 | 8 | 0 |
| Disk | 268,435,456 | 6214.4 MiB/s | 1.130 ms | 1.604 ms | 256 | 8 | 0 |
| Unity Catalog | 268,435,456 | 1148.0 MiB/s | 5.332 ms | 17.458 ms | 256 | 8 | 0 |

## Memory / Footprint

| Metric | Value |
| --- | --- |
| Bytes read | 268,435,456 per reader |
| Peak GPU memory | not measured |
| CPU RSS | not measured |
| Cache-resident footprint | not measured |
| Serving latency | not measured |

This benchmark is storage-reader evidence only. It does not measure model
TTFT, time-to-completion, GPU memory, CPU RSS, or serving cache footprint.

## Provenance

Sanitized evidence is committed beside this README:

- [`storage_benchmark.json`](storage_benchmark.json)
- [`databricks_run_status.json`](databricks_run_status.json)
