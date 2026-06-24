# Storage Benchmark Report

This folder is the standalone, human-readable entry point for current Cachet
storage-reader benchmark results. Dated subfolders are the report pages to read
and cite; each dated folder carries the compact sanitized JSON records behind
the report. The matching `../databricks/` folders remain as release-bundle
source mirrors and Databricks run-status indexes.

## Current Results

| Target | Databricks run | Source artifacts | Reader | Throughput | p50 latency | p95 latency | Errors |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| AWS g6/L4, `g6.8xlarge` | `948365719597221` | [`2026-06-21-g6-l4-storage-readers/`](2026-06-21-g6-l4-storage-readers/) | memory | 6531.4 MiB/s | 0.847 ms | 1.649 ms | 0 |
| AWS g6/L4, `g6.8xlarge` | `948365719597221` | [`2026-06-21-g6-l4-storage-readers/`](2026-06-21-g6-l4-storage-readers/) | disk | 6214.4 MiB/s | 1.130 ms | 1.604 ms | 0 |
| AWS g6/L4, `g6.8xlarge` | `948365719597221` | [`2026-06-21-g6-l4-storage-readers/`](2026-06-21-g6-l4-storage-readers/) | unity_catalog | 1148.0 MiB/s | 5.332 ms | 17.458 ms | 0 |

## Scope

These storage results cover Memory, Disk, and Unity Catalog reader behavior on
the strict AWS g6/L4 release target. They pair with the vLLM V1 benchmark
evidence, but they are not model-serving latency or quality measurements.

## Evidence Boundary

Use this folder for human review and citation. The dated report folder includes
the sanitized `document_kv.storage_benchmark.v1` and
`document_kv.databricks_run_status.v1` records it cites. Use `../databricks/`
as a release-source mirror, not as the only benchmark surface. Do not use
`../../pr-evidence/` as the benchmark report surface.
