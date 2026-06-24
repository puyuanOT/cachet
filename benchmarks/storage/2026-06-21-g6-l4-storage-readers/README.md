# AWS g6/L4 Storage Reader Benchmark

This standalone report records the current Memory, Disk, and Unity Catalog
storage-reader benchmark that pairs with the strict Cachet V1 release target.

| Field | Value |
| --- | --- |
| Date | 2026-06-21 |
| Scope | Storage reader benchmark |
| Target | `aws-g6-l4` / `g6.8xlarge` |
| Databricks run | `948365719597221` |
| Readers | `memory`, `disk`, `unity_catalog` |
| UC Volume | Real Unity Catalog Volume |
| Result | Published storage benchmark; `ok=true`, zero reader errors |

## Human Result

All three storage readers completed without errors on the strict g6/L4 target.
Memory and disk readers exceeded 6 GiB/s, and the Unity Catalog reader reached
1148.0 MiB/s against a real UC Volume.

| Reader | Throughput | p50 latency | p95 latency | Errors |
| --- | ---: | ---: | ---: | ---: |
| memory | 6531.4 MiB/s | 0.847 ms | 1.649 ms | 0 |
| disk | 6214.4 MiB/s | 1.130 ms | 1.604 ms | 0 |
| unity_catalog | 1148.0 MiB/s | 5.332 ms | 17.458 ms | 0 |

## Scope

This folder proves storage-reader behavior for Cachet's benchmark inputs. It
is not a model-serving latency, throughput, or quality benchmark.

## Source Artifacts

The sanitized source records are committed beside this README:

- `storage_benchmark.json`
- `databricks_run_status.json`

The same records are mirrored under
[`../../databricks/2026-06-21-g6-l4-storage-readers/`](../../databricks/2026-06-21-g6-l4-storage-readers/)
for release-bundle and Databricks run-status audits.

## Artifact Boundary

This folder is the human-readable benchmark report. Keep raw Databricks Jobs
API responses, tokens, wheels, logs, generated datasets, and local scratch
output out of this tree.
