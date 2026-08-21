# Storage Benchmark Index

Storage appears in the [benchmark root](../) as a storage-tier ablation and as
resource-utilization fields. The current RAM, local-disk, and mounted Unity
Catalog serving measurements are backed by the
[sanitized descriptive evidence](../appendix/main-vanilla-descriptive-evidence/).
The Unity Catalog backend-cache state is unproven, so that row is not labeled a
strict cold-UC measurement. Historical storage-reader-only results were removed
from `benchmarks/` because they did not measure model-serving latency under the
current Q4-weight + Q8-document-KV protocol.
