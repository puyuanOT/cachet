# Primary Table V2 Evidence: vLLM Qwen3 4B on g5/A10G

This folder contains superseded sanitized Databricks evidence from an earlier
primary-table pass. The current benchmark root main table uses
[`../primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/`](../primary-table-v4-vllm-qwen3-4b-g5-a10g-disk-cache-forced256-telemetry/),
which separates forced-256 latency from natural-stop quality and includes
provider telemetry. This V2 folder remains useful historical evidence, but it
should not be treated as the current main-table source.

## Configuration

| Field | Value |
| --- | --- |
| Model | Qwen3-4B-Instruct, `Qwen/Qwen3-4B-Instruct-2507` |
| Served model name | `qwen3:4b-instruct` |
| Serving engine | vLLM `0.23.0` |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for TTC | 256 generated tokens |
| vLLM capacity setting | `--max-model-len 32768`, `--max-num-seqs 8` |
| Dataset rows | One prepared synthetic row each for Biography, HotpotQA, MusiQue, and NIAH |
| Latency percentile scope | 32 successful request-level measurements per method/context pair |
| Main quality metric | Prepared-suite exact match (EM) |
| Secondary quality metric | `answer_found_rate` retained in raw evidence only |
| Cachet method | Vanilla external KV |
| Cache residency | Local disk under `/local_disk0`, not Unity Catalog |
| Cachet prompt mode | Logical prompt text with vLLM KV-transfer metadata |
| DBFS staging root | `dbfs:/benchmarks/cachet/primary-table-v2-fc43996-20260627_033434` |
| Source commit | `fc43996` |

## Result Table

Latency values are seconds. EM is exact-match rate over the prepared synthetic
row for that dataset and context. These EM values are not full-dataset
Biography, HotpotQA, MusiQue, or NIAH accuracy.

| Method | Input context | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | Biography EM | HotpotQA EM | MusiQue EM | NIAH EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | 8k | 2.74 | 10.64 | 21.34 | 29.23 | 1.00 | 0.00 | 0.00 | 0.00 |
| Baseline, no precomputed KV | 16k | 29.43 | 34.11 | 49.13 | 53.82 | 0.00 | 0.00 | 0.00 | 0.00 |
| Baseline, no precomputed KV | 32k | 96.17 | 97.35 | 114.66 | 114.97 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cachet + vanilla KV | 8k | 7.97 | 8.16 | 17.74 | 17.89 | 1.00 | 0.00 | 0.00 | 0.00 |
| Cachet + vanilla KV | 16k | 27.52 | 27.93 | 37.28 | 37.70 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cachet + vanilla KV | 32k | 57.77 | 62.20 | 71.35 | 71.78 | 0.00 | 0.00 | 0.00 | 0.00 |

## Interpretation

The Cachet rows validate the current vLLM external-KV handoff path: requests
attach Cachet handoff metadata, the vLLM native provider imports vanilla KV
blocks from local disk, and the request uses `prompt_text_mode=logical`.
Suffix-only runtime prompt text is not supported by this provider today, because
vLLM's connector lifecycle matches external KV against the request prefix.

The 32k Cachet TTFT remains high in absolute terms. This is an end-to-end
serving measurement that includes request handling, tokenization/accounting,
local-disk raw-KV materialization, payload view/merge work, and GPU KV-block
injection. It is not an isolated in-memory KV-load microbenchmark.

## Provenance

| File | Contents |
| --- | --- |
| [`summary.json`](summary.json) | Aggregated table rows, configuration, limitations, and Databricks run IDs |
| [`vllm_import_probe.json`](vllm_import_probe.json) | vLLM import/native-provider probe from the run environment |
| [`prompt_token_budget_8k.json`](prompt_token_budget_8k.json) | Tokenizer budget probe for 8k prepared inputs |
| [`prompt_token_budget_16k.json`](prompt_token_budget_16k.json) | Tokenizer budget probe for 16k prepared inputs |
| [`prompt_token_budget_32k.json`](prompt_token_budget_32k.json) | Tokenizer budget probe for 32k prepared inputs |
| [`prepared_handoff_generation_8k.json`](prepared_handoff_generation_8k.json) | Local-disk Cachet handoff generation summary for 8k Cachet rows |
| [`prepared_handoff_generation_16k.json`](prepared_handoff_generation_16k.json) | Local-disk Cachet handoff generation summary for 16k Cachet rows |
| [`prepared_handoff_generation_32k.json`](prepared_handoff_generation_32k.json) | Local-disk Cachet handoff generation summary for 32k Cachet rows |
| [`prepared_handoff_coverage_8k.json`](prepared_handoff_coverage_8k.json) | Handoff coverage validation for 8k Cachet rows |
| [`prepared_handoff_coverage_16k.json`](prepared_handoff_coverage_16k.json) | Handoff coverage validation for 16k Cachet rows |
| [`prepared_handoff_coverage_32k.json`](prepared_handoff_coverage_32k.json) | Handoff coverage validation for 32k Cachet rows |
| [`v1_benchmark_8k_baseline.json`](v1_benchmark_8k_baseline.json) | Raw benchmark record for the 8k baseline row |
| [`v1_benchmark_8k_cachet_vanilla_kv.json`](v1_benchmark_8k_cachet_vanilla_kv.json) | Raw benchmark record for the 8k Cachet + vanilla KV row |
| [`v1_benchmark_16k_baseline.json`](v1_benchmark_16k_baseline.json) | Raw benchmark record for the 16k baseline row |
| [`v1_benchmark_16k_cachet_vanilla_kv.json`](v1_benchmark_16k_cachet_vanilla_kv.json) | Raw benchmark record for the 16k Cachet + vanilla KV row |
| [`v1_benchmark_32k_baseline.json`](v1_benchmark_32k_baseline.json) | Raw benchmark record for the 32k baseline row |
| [`v1_benchmark_32k_cachet_vanilla_kv.json`](v1_benchmark_32k_cachet_vanilla_kv.json) | Raw benchmark record for the 32k Cachet + vanilla KV row |

Databricks run IDs: `858090300148378`, `565022243520261`,
`20555720374513`, `824809886933485`, `128215351555027`, and
`683759070768615`.
