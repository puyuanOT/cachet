# Storage Benchmark Index

Storage appears in the [benchmark root](../) as a storage-tier ablation and as
resource-utilization fields. Historical storage-reader-only results were
removed from `benchmarks/` because they did not measure model-serving latency
under the current Q4-weight + Q8-document-KV protocol.
