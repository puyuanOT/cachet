# Storage Benchmarks

These results measure Cachet storage-reader throughput and latency. They are
not model-serving latency benchmarks and are not memory-consumption
measurements.

## Experimental Setup

| Result | Hardware | Method | Workload | Repeats / measurements | Evidence |
| --- | --- | --- | --- | --- | --- |
| [`g6-l4-reader-throughput/`](g6-l4-reader-throughput/) | AWS g6/L4, `g6.8xlarge` | Memory, disk, Unity Catalog readers | 268,435,456 total bytes per reader | 256 reads; parallelism 8 | [`g6-l4-reader-throughput/storage_benchmark.json`](g6-l4-reader-throughput/storage_benchmark.json) |

## Reader Throughput And Latency

| Reader | Total bytes | Throughput | p50 latency | p95 latency | Reads | Parallelism | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Memory | 268,435,456 | 6531.4 MiB/s | 0.847 ms | 1.649 ms | 256 | 8 | 0 |
| Disk | 268,435,456 | 6214.4 MiB/s | 1.130 ms | 1.604 ms | 256 | 8 | 0 |
| Unity Catalog | 268,435,456 | 1148.0 MiB/s | 5.332 ms | 17.458 ms | 256 | 8 | 0 |

## Memory / Footprint Interpretation

| Metric | Value | Notes |
| --- | --- | --- |
| Bytes read | 268,435,456 per reader | Storage workload size, not resident memory |
| Peak GPU memory | not measured | Storage benchmark does not run model serving |
| CPU RSS | not measured | Not present in the artifact |
| Cache-resident footprint | not measured | Not present in the artifact |
| Serving TTFT / TTC | not measured | Use [`../vllm/`](../vllm/) or [`../sglang/`](../sglang/) |

## Coverage

| Reader | Hardware | Status |
| --- | --- | --- |
| Memory | AWS g6/L4 | Benchmark passed with zero errors |
| Disk | AWS g6/L4 | Benchmark passed with zero errors |
| Unity Catalog Volume | AWS g6/L4 | Benchmark passed with zero errors |
| Other storage backends | Any hardware | not benchmarked yet |
