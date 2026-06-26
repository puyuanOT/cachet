# SGLang Qwen3 4B On g6/L4 With Vanilla KV

Appendix evidence for a prepared live SGLang benchmark through HiCache. This
validates correctness and cache-hit integration; it does not demonstrate a
latency improvement and is not a row in the primary table in
[benchmark root](../../../).

## Experimental Setup

| Field | Value |
| --- | --- |
| Engine | SGLang |
| Model | `qwen3:4b-instruct` |
| Hardware | AWS g6/L4, `g6.8xlarge` |
| Method | Vanilla external KV via HiCache |
| Baseline arm | `baseline_prefill` |
| Cache arm | `document_kv_cache` |
| Dataset scope | Biography, HotpotQA, MusiQue, NIAH |
| Repeats / measurements | 2 repeats per arm/dataset; 16 measurements |
| Prompt-token scope | report-row prompt mean not recorded; cache-validation prompt tokens 120-189 |
| Evidence file | [`success_run.json`](success_run.json) |
| Primary-table mismatch | SGLang on g6/L4, 2 repeats, short prompts, and no fixed parallel-8/256-token/g5/disk-cache configuration |

## Main Latency Results

| Dataset | Baseline p50 TTFT | Cachet p50 TTFT | TTFT speedup | Baseline p50 TTC | Cachet p50 TTC | TTC speedup | Validated cached tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| biography | 0.197 s | 0.204 s | 0.966x | 0.727 s | 0.753 s | 0.966x | 96 |
| hotpotqa | 0.081 s | 0.257 s | 0.314x | 1.412 s | 1.585 s | 0.891x | 144 |
| musique | 0.081 s | 0.225 s | 0.358x | 1.416 s | 1.557 s | 0.910x | 144 |
| niah | 0.077 s | 0.245 s | 0.313x | 1.410 s | 1.576 s | 0.895x | 96 |

Speedup below `1.0x` means Cachet was slower than baseline on these short
prepared prompts.

## Quality Results

| Dataset | Baseline `answer_found_rate` | Cachet `answer_found_rate` | `answer_found_rate` delta | Baseline `exact_match_rate` | Cachet `exact_match_rate` | `exact_match_rate` delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| biography | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hotpotqa | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| musique | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| niah | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Memory / Footprint

| Metric | Value |
| --- | --- |
| Cache-hit validations | 8/8 passed |
| Validated cached tokens | 96-144 |
| Peak GPU memory | Not measured |
| CPU RSS | Not measured |
| Cache-resident footprint | Not measured |
| Storage throughput | Not measured in this serving benchmark |

## Limitations

| Limitation | Current state |
| --- | --- |
| Performance | No latency improvement observed on the current short prepared prompts |
| Repeat count | 2 repeats per arm/dataset |
| Prompt tokens | report-row prompt means are not recorded in this artifact |
| Memory | Serving peak GPU memory, CPU RSS, and KV cache footprint are not measured |

## Provenance

[`success_run.json`](success_run.json) contains the sanitized terminal run state,
handoff generation, measurements, comparisons, and cache-hit validations.
