# vLLM Benchmark Report

This folder is the standalone, human-readable entry point for current Cachet
vLLM latency and quality benchmark results. Dated subfolders are the report
pages to read and cite; the canonical sanitized JSON artifacts remain under
`../databricks/` so release validation can audit the same records cited here.

## Current Results

| Target | Databricks run | Source artifacts | Measurements | TTFT speedup | Time-to-completion speedup | Quality delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Strict AWS g6/L4, `g6.8xlarge` | `872615985402004` | [`2026-06-23-g6-l4-v1/`](2026-06-23-g6-l4-v1/) | 24 | 5.27x-6.97x | 1.74x-2.25x | 0.0 |
| Compatibility AWS g5/A10G, `g5.8xlarge` | `566743786103032` | [`2026-06-23-g5-a10g-v1-compatibility/`](2026-06-23-g5-a10g-v1-compatibility/) | 24 | 4.66x-6.04x | 2.04x-2.67x | 0.0 |

## Scope

These are the currently published Cachet V1 latency and quality benchmark
results. They compare the `baseline_prefill` arm with the `document_kv_cache`
arm on Qwen3 4B Instruct across Biography, HotpotQA, MusiQue, and NIAH.

The strict publication target is AWS g6/L4 on plain `g6.8xlarge` Databricks
hardware. The g5/A10G row is compatibility evidence and cannot replace the
strict g6/L4 release target.

## Evidence Boundary

Use this folder for human review and citation. Use the linked Databricks folders
for the sanitized `document_kv.benchmark_run.v1`,
`document_kv.databricks_run_status.v1`, and release-evidence source records.
Do not use `../../pr-evidence/` as the benchmark report surface.
