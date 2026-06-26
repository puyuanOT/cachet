# SGLang Synthetic NIAH With Vanilla KV

Appendix evidence for repeated Cachet-backed cache hits on one synthetic NIAH
prompt. It is useful integration evidence, not the full SGLang release-suite
benchmark and not a row in the fixed main table in
[`../../../current/`](../../../current/).

## Experimental Setup

| Field | Value |
| --- | --- |
| Engine | SGLang |
| Model | `qwen3:4b-instruct` |
| Hardware | AWS g6/L4, `g6.8xlarge` |
| Method | Vanilla external KV via HiCache |
| Baseline arm | `baseline_prefill` |
| Cache arm | `document_kv_cache` |
| Dataset scope | One synthetic NIAH prompt |
| Repeats / measurements | 2 repeats per arm; 4 measurements |
| Prompt-token scope | report-row mean prompt tokens 92; cache-validation prompt tokens 205 |
| Evidence file | [`success_run.json`](success_run.json) |
| Main-table mismatch | One tiny synthetic prompt on SGLang/g6/L4, not the fixed vLLM/g5/parallel-8/256-token/four-dataset table |

## Main Latency Results

| Scope | Baseline p50 TTFT | Cachet p50 TTFT | TTFT speedup | Baseline p50 TTC | Cachet p50 TTC | TTC speedup | Validated cached tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic NIAH | 0.303 s | 0.346 s | 0.875x | 0.556 s | 0.601 s | 0.926x | 175 |

## Quality Results

| Scope | Baseline answer-found | Cachet answer-found | Answer-found delta | Baseline exact-match | Cachet exact-match | Exact-match delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic NIAH | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |

## Memory / Footprint

| Metric | Value |
| --- | --- |
| Cache-hit validations | 2/2 passed |
| Validated cached tokens | 175 |
| Peak GPU memory | not measured |
| CPU RSS | not measured |
| Cache-resident footprint | not measured |
| Storage throughput | not measured in this serving benchmark |

## Limitations

| Limitation | Current state |
| --- | --- |
| Dataset scope | One synthetic prompt |
| Performance | No speedup on this tiny prompt |
| Repeat count | 2 repeats per arm |
| Memory | Serving peak GPU memory, CPU RSS, and cache footprint are not measured |

## Provenance

[`success_run.json`](success_run.json) contains the sanitized terminal run state,
smoke gates, live benchmark rows, comparisons, and cache-hit validations.
