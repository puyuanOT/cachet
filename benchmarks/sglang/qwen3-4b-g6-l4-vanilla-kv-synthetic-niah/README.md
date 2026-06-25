# SGLang Synthetic NIAH With Vanilla KV

This small SGLang benchmark validates repeated live Cachet-backed cache hits on
one synthetic NIAH prompt. It is useful integration evidence, but it is not the
full SGLang release-suite benchmark.

| Field | Value |
| --- | --- |
| Serving platform | SGLang |
| Model | `qwen3:4b-instruct` |
| Hardware | AWS g6/L4, `g6.8xlarge` |
| Method | Vanilla external KV through SGLang HiCache |
| Scope | One synthetic NIAH prompt |
| Repeats | Two baseline repeats, two Cachet cache-arm repeats |
| Cache-hit validation | Both Cachet repeats validated 175 cached tokens |
| Quality | answer-found delta `0.0` |
| Result | Correct cache hits; no speedup on this tiny prompt |

## Result

Baseline p50 TTFT was `0.3029023165s`; Cachet p50 TTFT was
`0.3463493005s`, for `0.875x` TTFT speedup. Baseline p50
time-to-completion was `0.5560753460s`; Cachet p50 time-to-completion was
`0.6006555025s`, for `0.926x` speedup.

## Provenance

`success_run.json` contains the sanitized terminal run state, smoke gates, live
benchmark rows, comparisons, and cache-hit validations.
