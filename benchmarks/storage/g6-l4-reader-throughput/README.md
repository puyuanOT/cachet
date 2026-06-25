# Storage Reader Throughput On g6/L4

This benchmark measures Cachet reader performance for memory, disk, and real
Unity Catalog Volume storage on the g6/L4 target.

| Reader | Throughput | p50 Latency | p95 Latency | Errors |
| --- | ---: | ---: | ---: | ---: |
| Memory | 6531.4 MiB/s | 0.847 ms | 1.649 ms | 0 |
| Disk | 6214.4 MiB/s | 1.130 ms | 1.604 ms | 0 |
| Unity Catalog | 1148.0 MiB/s | 5.332 ms | 17.458 ms | 0 |

## Scope

This is storage-reader evidence, not model-serving latency evidence. It pairs
with the vLLM and SGLang serving benchmarks by showing that Cachet's storage
path can retrieve KV payload bytes from supported backends.

## Provenance

Sanitized evidence is committed beside this README:

- `storage_benchmark.json`
- `databricks_run_status.json`
