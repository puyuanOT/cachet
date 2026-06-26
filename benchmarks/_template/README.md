# Benchmark Report Template

Use this template for new public benchmark result folders under `benchmarks/`.
Folder names should be stable and descriptive, such as
`qwen3-4b-g6-l4-vanilla-kv`, not date or run-id based.

Do not invent missing numbers. If a metric is absent from committed evidence,
write `not measured`, `not recorded`, `n/a`, or `not benchmarked yet`.

## Experimental Setup

| Field | Value |
| --- | --- |
| Engine | vLLM, SGLang, storage reader, native probe, etc. |
| Model | e.g. `qwen3:4b-instruct`, or `n/a` |
| Hardware | e.g. AWS g6/L4, `g6.8xlarge` |
| Method | Vanilla external KV, KV Packet, storage reader, etc. |
| Baseline arm | e.g. `baseline_prefill`, reader baseline, or `n/a` |
| Cache arm | e.g. `document_kv_cache`, provider-backed connector, or `n/a` |
| Dataset scope | Dataset names and example count |
| Repeats / measurements | Repeat count and total measurement count |
| Prompt-token scope | Range or mean if present; otherwise `not recorded` |
| Evidence file | Link to sanitized committed JSON |

## Main Latency Results

Speedup should be `baseline p50 / Cachet p50`. Values below `1.0x` mean
Cachet was slower.

| Dataset or scope | Baseline p50 TTFT | Cachet p50 TTFT | TTFT speedup | Baseline p50 time-to-completion | Cachet p50 time-to-completion | TTC speedup | Repeats | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| example dataset | `not measured` | `not measured` | `not measured` | `not measured` | `not measured` | `not measured` | `not recorded` | State whether this is speedup, compatibility, correctness, storage, or integration evidence |

## Quality Results

Define the quality metric before reporting it. Current Cachet benchmark docs use
`answer_found_rate` as the main quality gate when present. Exact match is a
separate stricter raw field.

| Dataset or scope | Baseline answer-found | Cachet answer-found | Answer-found delta | Baseline exact-match | Cachet exact-match | Exact-match delta | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| example dataset | `not measured` | `not measured` | `not measured` | `not measured` | `not measured` | `not measured` | Explain any metric caveat |

## Memory / Footprint

Do not use storage throughput as a synonym for memory consumption. Report the
footprint fields that are present and mark missing serving-memory metrics.

| Evidence | Bytes | Throughput | p50 latency | Copied tokens | Bytes per token | Estimated GPU bytes | Peak GPU memory | CPU RSS | Cache footprint | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| example | `not measured` | `not measured` | `not measured` | `not measured` | `not measured` | `not measured` | `not measured` | `not measured` | `not measured` | State what the metric does and does not prove |

## Coverage Matrix

| Engine | Model | Hardware | Method | Status |
| --- | --- | --- | --- | --- |
| example | example | example | example | `benchmarked`, `integration evidence`, `not measured`, or `not benchmarked yet` |

## Limitations

Keep this short and honest. Include the relevant rows:

| Limitation | Current state |
| --- | --- |
| Model coverage | e.g. one model only, or list covered models |
| Method coverage | e.g. vanilla external KV only; KV Packet not benchmarked yet |
| Repeat count | e.g. 3 repeats per arm/dataset |
| Memory | e.g. peak GPU memory and CPU RSS not measured |
| Serving scope | e.g. storage benchmark is not serving latency |

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
