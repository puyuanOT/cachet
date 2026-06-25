# Storage Benchmarks

These results measure Cachet storage-reader throughput and latency. They are
not model-serving benchmarks, but they explain whether Cachet can read KV
payload bytes quickly enough from supported storage backends.

## Results

| Result | Hardware | Reader | Throughput | p50 Latency | p95 Latency | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [`g6-l4-reader-throughput/`](g6-l4-reader-throughput/) | AWS g6/L4, `g6.8xlarge` | Memory | 6531.4 MiB/s | 0.847 ms | 1.649 ms | 0 |
| [`g6-l4-reader-throughput/`](g6-l4-reader-throughput/) | AWS g6/L4, `g6.8xlarge` | Disk | 6214.4 MiB/s | 1.130 ms | 1.604 ms | 0 |
| [`g6-l4-reader-throughput/`](g6-l4-reader-throughput/) | AWS g6/L4, `g6.8xlarge` | Unity Catalog | 1148.0 MiB/s | 5.332 ms | 17.458 ms | 0 |

## What This Proves

The current storage layer supports memory, local disk, and real Unity Catalog
Volume readers with zero reader errors on the g6/L4 benchmark target.

Use [`../vllm/`](../vllm/) or [`../sglang/`](../sglang/) for serving latency
and quality comparisons.
