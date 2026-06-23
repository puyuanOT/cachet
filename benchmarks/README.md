# Benchmark Evidence

This directory contains curated Cachet benchmark evidence. It is separate from
`pr-evidence/`, which records PR review and validation history, and from
`databricks-runs/`, which remains ignored local scratch output.

Start with the current Databricks snapshot:
[`databricks/CURRENT.md`](databricks/CURRENT.md). Each benchmark folder contains
a human-readable `README.md` plus the sanitized JSON artifacts needed to
reproduce the claim. Do not commit Databricks tokens, raw Jobs API responses,
package wheels, logs, generated dataset payloads, or cluster-local scratch
output here.

## Current Databricks Evidence

| Folder | Purpose | Databricks run | Target | Result |
| --- | --- | --- | --- | --- |
| [`databricks/2026-06-23-g6-l4-v1`](databricks/2026-06-23-g6-l4-v1/) | Strict V1 benchmark target | `872615985402004` | `aws-g6-l4` / `g6.8xlarge` | `ok=true`; 24 measurements; 5.27x-6.97x TTFT speedups |
| [`databricks/2026-06-23-g5-a10g-v1-compatibility`](databricks/2026-06-23-g5-a10g-v1-compatibility/) | Non-default compatibility benchmark | `566743786103032` | `aws-g5-a10g` / `g5.8xlarge` | `ok=true`; 24 measurements; 4.66x-6.04x TTFT speedups |
| [`databricks/2026-06-21-g6-l4-storage-readers`](databricks/2026-06-21-g6-l4-storage-readers/) | Memory, Disk, and Unity Catalog storage readers | `948365719597221` | `aws-g6-l4` / `g6.8xlarge` | `ok=true`; real UC Volume; zero reader errors |
| [`databricks/2026-06-23-g6-l4-native-engine-probes`](databricks/2026-06-23-g6-l4-native-engine-probes/) | vLLM and SGLang native connector probes | `934698284395881` | `aws-g6-l4` / `g6.8xlarge` | `ok=true`; provider-backed native probes succeeded |

The strict V1 publication target remains AWS g6/L4. The g5 folder is retained
only as compatibility evidence and cannot replace the g6/L4 release target.
