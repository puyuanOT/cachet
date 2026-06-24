# Benchmark Report Template

Use this template for each standalone benchmark or benchmark-readiness folder
under `benchmarks/`. The folder should be readable without opening
`pr-evidence/`, raw Databricks job payloads, or local `databricks-runs/`
scratch output.

## Summary

| Field | Value |
| --- | --- |
| Date | YYYY-MM-DD |
| Scope | vLLM V1, SGLang live V1, storage readers, native engine probe, or readiness smoke |
| Target | `aws-g6-l4` / `g6.8xlarge` unless explicitly marked compatibility |
| Databricks run | RUN_ID |
| Result | `ok=true`, `ok=false`, or readiness-only status |
| Publication status | Published benchmark, compatibility evidence, integration evidence, or readiness evidence |

## Human Result

State the benchmark claim in a few sentences. Include the useful numbers a
reviewer should see first, such as measurement count, TTFT speedup,
time-to-completion speedup, quality delta, cache-hit validation, reader
throughput, or probe copied-token count.

## Scope

Explain exactly what this folder proves and what it does not prove. For
example, native connector probes are integration evidence, synthetic SGLang
live measurements are not full V1 release-suite results, and g5/A10G folders
are compatibility evidence rather than the strict g6/L4 publication target.

## Source Artifacts

List the sanitized records committed beside this README, such as:

- `v1_benchmark.json`
- `sglang-live-benchmark.json`
- `storage_benchmark.json`
- `databricks_run_status.json`
- `release_evidence.json`
- `*_engine_probe.json`
- `*_connector_actions.json`

## Artifact Boundary

Do not include Databricks tokens, raw Jobs API responses, package wheels,
cluster logs, generated payload blobs, or local scratch directories. Use
`pr-evidence/` only for PR validation and release-audit sidecars, not as the
human benchmark report surface.
