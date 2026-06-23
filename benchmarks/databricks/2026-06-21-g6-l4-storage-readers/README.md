# AWS g6/L4 Storage Reader Benchmark

This folder records the current Memory, Disk, and Unity Catalog storage-reader
evidence that pairs with the strict V1 release benchmark.

| Field | Value |
| --- | --- |
| Databricks run | `948365719597221` |
| Run name | `cachet-storage-benchmark-20260621_095026` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Readers | `memory`, `disk`, `unity_catalog` |
| UC Volume | Real Unity Catalog Volume |
| Result | `ok=true`, no reader errors |

## Results

| Reader | Throughput | p50 latency | p95 latency | Errors |
| --- | ---: | ---: | ---: | ---: |
| memory | 6531.4 MiB/s | 0.847 ms | 1.649 ms | 0 |
| disk | 6214.4 MiB/s | 1.130 ms | 1.604 ms | 0 |
| unity_catalog | 1148.0 MiB/s | 5.332 ms | 17.458 ms | 0 |

## Artifacts

- [`storage_benchmark.json`](storage_benchmark.json) contains the
  `document_kv.storage_benchmark.v1` storage benchmark report.
- [`databricks_run_status.json`](databricks_run_status.json) contains the
  terminal successful Databricks run-status summary.
