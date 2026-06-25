# Benchmark Report Template

Use this template for new public benchmark result folders under `benchmarks/`.
Folder names should be stable and descriptive, such as
`qwen3-4b-g6-l4-vanilla-kv`, not date or run-id based.

## Summary

| Field | Value |
| --- | --- |
| Serving platform | vLLM, SGLang, storage, or native engine probe |
| Model | e.g. `qwen3:4b-instruct`, or `n/a` for storage |
| Hardware | e.g. AWS g6/L4, `g6.8xlarge` |
| Method | Vanilla external KV, KV Packet, storage reader, etc. |
| Baseline | no-cache prefill, reader baseline, or `n/a` |
| Measurements | Count and scope |
| Result | Speedup, no speedup, correctness-only, or integration-only |

## Result

State the useful benchmark claim first. Include the numbers readers care about:
TTFT speedup, time-to-completion speedup, throughput, quality delta, cache-hit
validation, copied-token count, or reader errors.

## Scope

Explain what this folder proves and what it does not prove. Examples:

- g5/A10G is compatibility evidence, not the primary g6/L4 target.
- SGLang correctness/cache-hit evidence is not automatically a speedup claim.
- Native connector probes are integration evidence, not latency benchmarks.
- Planned methods such as KV Packet need their own result folders.

## Provenance

List sanitized records committed beside this README, such as:

- `v1_benchmark.json`
- `success_run.json`
- `storage_benchmark.json`
- `databricks_run_status.json`
- `release_evidence.json`
- `*_engine_probe.json`
- `*_connector_actions.json`

Do not include Databricks tokens, raw Jobs API responses, package wheels,
cluster logs, generated payload blobs, prompt text, or local scratch
directories.
