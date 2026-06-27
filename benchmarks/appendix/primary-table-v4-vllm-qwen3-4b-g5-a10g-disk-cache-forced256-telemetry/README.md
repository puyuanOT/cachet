# Primary Table V4 Evidence: vLLM Qwen3 4B on g5/A10G

This folder contains the sanitized Databricks evidence used by the benchmark root main table. It separates forced-256 latency measurements from natural-stop quality measurements and includes provider telemetry for Cachet vanilla KV loads.

## Configuration

| Field | Value |
| --- | --- |
| Model | Qwen3-4B-Instruct, `Qwen/Qwen3-4B-Instruct-2507` |
| Served model name | `qwen3:4b-instruct` |
| Serving engine | vLLM `0.23.0` |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Latency decode protocol | `max_tokens=256`, `ignore_eos=true`; every successful latency measurement emitted 256 completion tokens |
| Quality protocol | Natural EOS with `max_tokens=256`; exact match from these runs populates score columns |
| vLLM capacity setting | `--max-model-len 32768`, `--max-num-seqs 8` |
| Dataset rows | One prepared synthetic row each for Biography, HotpotQA, MusiQue, and NIAH |
| Latency percentile scope | 32 successful request-level measurements per method/context pair |
| Main quality metric | Prepared-suite exact match (EM) |
| Secondary quality metric | `answer_found_rate` retained in raw evidence only |
| Cachet method | Vanilla external KV |
| Cache residency | Local disk under `/local_disk0`, not Unity Catalog |
| Cachet prompt mode | Logical prompt text with vLLM KV-transfer metadata |
| Latency DBFS staging root | `dbfs:/benchmarks/cachet/primary-table-v4-forced256-telemetry-20260627_063513` |
| Quality DBFS staging root | `dbfs:/benchmarks/cachet/primary-table-v3-telemetry-20260627_061645` |

## Result Table

Latency values are seconds. TTFT and TTC come from forced-256 latency runs. EM comes from natural-stop quality runs over the same prepared examples.

| Method | Input context | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography EM | HotpotQA EM | MusiQue EM | NIAH EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | 8k | 2.73 | 9.89 | 21.38 | 28.51 | 1.00 | 0.00 | 0.00 | 0.00 |
| Baseline, no precomputed KV | 16k | 29.43 | 34.12 | 49.13 | 53.83 | 0.00 | 0.00 | 0.00 | 0.00 |
| Baseline, no precomputed KV | 32k | 96.13 | 97.31 | 114.61 | 114.93 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cachet + vanilla KV | 8k | 7.91 | 8.17 | 17.74 | 17.98 | 1.00 | 0.00 | 0.00 | 0.00 |
| Cachet + vanilla KV | 16k | 27.54 | 28.03 | 37.31 | 37.79 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cachet + vanilla KV | 32k | 57.71 | 62.08 | 71.28 | 71.67 | 0.00 | 0.00 | 0.00 | 0.00 |

## Cachet Connector Telemetry

The Cachet rows wrote one provider-load telemetry record per successful request. `payload materialize` is local-disk payload read/materialization; `GPU layer load` is the provider copy into vLLM KV-cache tensors.

| Input context | Records | Cached tokens p50 | Payload p50 | Provider load p50 / p95 | Payload materialize p50 / p95 | GPU layer load p50 / p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8k | 32 | 8160 | 1.12 GiB | 0.85 / 0.93 s | 0.66 / 0.75 s | 0.18 / 0.18 s |
| 16k | 32 | 16656 | 2.29 GiB | 1.95 / 2.12 s | 1.35 / 1.52 s | 0.59 / 0.61 s |
| 32k | 32 | 31432 | 4.32 GiB | 3.63 / 3.93 s | 2.55 / 2.86 s | 1.07 / 1.09 s |

## Interpretation

The Cachet rows validate the current vLLM external-KV handoff path: requests attach Cachet handoff metadata, vLLM reports near-100% external prefix cache hit rate in server logs for Cachet rows, and provider telemetry records 32 successful local-disk vanilla-KV loads per context length. The request still uses `prompt_text_mode=logical`; suffix-only runtime prompt text is not supported by this provider today because vLLM matches external KV against a request prefix.

Baseline rows use the same connector-enabled vLLM server for parity, but
baseline requests do not attach Cachet KV-transfer parameters; server evidence
reports 0% external-prefix cache hits for those rows and no baseline provider
telemetry load files are committed.

The 32k Cachet TTFT is high for an implementation that skips cached-token prefill, but the evidence shows why: provider load for the 32k row is about 3.63s p50 / 3.93s p95 for a 4.32 GiB raw-KV payload, while vLLM only reports 2.53x maximum concurrency for 32,768-token requests on this g5.8xlarge configuration. Under an 8-way client load, requests queue in waves and still reserve GPU KV cache for the full logical context.

## Provenance

| File | Contents |
| --- | --- |
| [`summary.json`](summary.json) | Aggregated table rows, configuration, limitations, run IDs, and telemetry summaries |
| [`server_log_summary.json`](server_log_summary.json) | Extracted vLLM server capacity/cache-hit/queue summaries; raw server logs are not committed |
| [`vllm_import_probe.json`](vllm_import_probe.json) | vLLM import/native-provider probe from the forced-latency run environment |
| [`prompt_token_budget_8k.json`](prompt_token_budget_8k.json) | Tokenizer budget probe for 8k prepared inputs |
| [`prepared_handoff_generation_8k.json`](prepared_handoff_generation_8k.json) | Local-disk Cachet handoff generation summary for 8k Cachet rows |
| [`prepared_handoff_coverage_8k.json`](prepared_handoff_coverage_8k.json) | Handoff coverage validation for 8k Cachet rows |
| [`connector_telemetry_8k_cachet_vanilla_kv.jsonl`](connector_telemetry_8k_cachet_vanilla_kv.jsonl) | Per-request provider load telemetry for 8k Cachet latency row |
| [`latency_v1_benchmark_8k_baseline.json`](latency_v1_benchmark_8k_baseline.json) | Forced-256 latency benchmark record for 8k baseline |
| [`quality_v1_benchmark_8k_baseline.json`](quality_v1_benchmark_8k_baseline.json) | Natural-stop quality benchmark record for 8k baseline |
| [`latency_metadata_8k_baseline.json`](latency_metadata_8k_baseline.json) | Forced-256 run metadata for 8k baseline |
| [`quality_metadata_8k_baseline.json`](quality_metadata_8k_baseline.json) | Natural-stop run metadata for 8k baseline |
| [`latency_v1_benchmark_8k_cachet_vanilla_kv.json`](latency_v1_benchmark_8k_cachet_vanilla_kv.json) | Forced-256 latency benchmark record for 8k Cachet + vanilla KV |
| [`quality_v1_benchmark_8k_cachet_vanilla_kv.json`](quality_v1_benchmark_8k_cachet_vanilla_kv.json) | Natural-stop quality benchmark record for 8k Cachet + vanilla KV |
| [`latency_metadata_8k_cachet_vanilla_kv.json`](latency_metadata_8k_cachet_vanilla_kv.json) | Forced-256 run metadata for 8k Cachet + vanilla KV |
| [`quality_metadata_8k_cachet_vanilla_kv.json`](quality_metadata_8k_cachet_vanilla_kv.json) | Natural-stop run metadata for 8k Cachet + vanilla KV |
| [`prompt_token_budget_16k.json`](prompt_token_budget_16k.json) | Tokenizer budget probe for 16k prepared inputs |
| [`prepared_handoff_generation_16k.json`](prepared_handoff_generation_16k.json) | Local-disk Cachet handoff generation summary for 16k Cachet rows |
| [`prepared_handoff_coverage_16k.json`](prepared_handoff_coverage_16k.json) | Handoff coverage validation for 16k Cachet rows |
| [`connector_telemetry_16k_cachet_vanilla_kv.jsonl`](connector_telemetry_16k_cachet_vanilla_kv.jsonl) | Per-request provider load telemetry for 16k Cachet latency row |
| [`latency_v1_benchmark_16k_baseline.json`](latency_v1_benchmark_16k_baseline.json) | Forced-256 latency benchmark record for 16k baseline |
| [`quality_v1_benchmark_16k_baseline.json`](quality_v1_benchmark_16k_baseline.json) | Natural-stop quality benchmark record for 16k baseline |
| [`latency_metadata_16k_baseline.json`](latency_metadata_16k_baseline.json) | Forced-256 run metadata for 16k baseline |
| [`quality_metadata_16k_baseline.json`](quality_metadata_16k_baseline.json) | Natural-stop run metadata for 16k baseline |
| [`latency_v1_benchmark_16k_cachet_vanilla_kv.json`](latency_v1_benchmark_16k_cachet_vanilla_kv.json) | Forced-256 latency benchmark record for 16k Cachet + vanilla KV |
| [`quality_v1_benchmark_16k_cachet_vanilla_kv.json`](quality_v1_benchmark_16k_cachet_vanilla_kv.json) | Natural-stop quality benchmark record for 16k Cachet + vanilla KV |
| [`latency_metadata_16k_cachet_vanilla_kv.json`](latency_metadata_16k_cachet_vanilla_kv.json) | Forced-256 run metadata for 16k Cachet + vanilla KV |
| [`quality_metadata_16k_cachet_vanilla_kv.json`](quality_metadata_16k_cachet_vanilla_kv.json) | Natural-stop run metadata for 16k Cachet + vanilla KV |
| [`prompt_token_budget_32k.json`](prompt_token_budget_32k.json) | Tokenizer budget probe for 32k prepared inputs |
| [`prepared_handoff_generation_32k.json`](prepared_handoff_generation_32k.json) | Local-disk Cachet handoff generation summary for 32k Cachet rows |
| [`prepared_handoff_coverage_32k.json`](prepared_handoff_coverage_32k.json) | Handoff coverage validation for 32k Cachet rows |
| [`connector_telemetry_32k_cachet_vanilla_kv.jsonl`](connector_telemetry_32k_cachet_vanilla_kv.jsonl) | Per-request provider load telemetry for 32k Cachet latency row |
| [`latency_v1_benchmark_32k_baseline.json`](latency_v1_benchmark_32k_baseline.json) | Forced-256 latency benchmark record for 32k baseline |
| [`quality_v1_benchmark_32k_baseline.json`](quality_v1_benchmark_32k_baseline.json) | Natural-stop quality benchmark record for 32k baseline |
| [`latency_metadata_32k_baseline.json`](latency_metadata_32k_baseline.json) | Forced-256 run metadata for 32k baseline |
| [`quality_metadata_32k_baseline.json`](quality_metadata_32k_baseline.json) | Natural-stop run metadata for 32k baseline |
| [`latency_v1_benchmark_32k_cachet_vanilla_kv.json`](latency_v1_benchmark_32k_cachet_vanilla_kv.json) | Forced-256 latency benchmark record for 32k Cachet + vanilla KV |
| [`quality_v1_benchmark_32k_cachet_vanilla_kv.json`](quality_v1_benchmark_32k_cachet_vanilla_kv.json) | Natural-stop quality benchmark record for 32k Cachet + vanilla KV |
| [`latency_metadata_32k_cachet_vanilla_kv.json`](latency_metadata_32k_cachet_vanilla_kv.json) | Forced-256 run metadata for 32k Cachet + vanilla KV |
| [`quality_metadata_32k_cachet_vanilla_kv.json`](quality_metadata_32k_cachet_vanilla_kv.json) | Natural-stop run metadata for 32k Cachet + vanilla KV |

Forced-latency Databricks run IDs: 907379196359027, 213179643238938, 924615285122791, 28545418549886, 157549365777924, 467713427964851.
Natural-stop quality Databricks run IDs: 387389783984285, 752674938892847, 531051381074166, 840810813746945, 87656226306908, 338179561067726.
