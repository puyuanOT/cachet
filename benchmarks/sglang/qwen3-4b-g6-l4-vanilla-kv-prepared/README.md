# SGLang Qwen3 4B On g6/L4 With Vanilla KV

This is the current prepared SGLang live benchmark for Cachet. It proves the
SGLang serving path can consume Cachet-backed external KV through HiCache with
correct answers and validated cache hits. It is not a speedup claim.

| Field | Value |
| --- | --- |
| Serving platform | SGLang |
| Model | `qwen3:4b-instruct` |
| Hardware | AWS g6/L4, `g6.8xlarge` |
| Method | Vanilla external KV through SGLang HiCache |
| Baseline | no-cache prefill |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Measurements | 16 |
| Cache-hit validations | 8/8 Cachet-backed cache arms passed |
| Quality | answer-found delta `0.0` on every dataset |
| Result | Correct and integrated, but slower than baseline on short prompts |

## Result

| Dataset | Baseline p50 TTFT | Cache p50 TTFT | TTFT Speedup | Baseline p50 TTC | Cache p50 TTC | TTC Speedup | Validated Cached Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Biography | `0.197s` | `0.204s` | `0.966x` | `0.727s` | `0.753s` | `0.966x` | `96` |
| HotpotQA | `0.081s` | `0.257s` | `0.314x` | `1.412s` | `1.585s` | `0.891x` | `144` |
| MusiQue | `0.081s` | `0.225s` | `0.358x` | `1.416s` | `1.557s` | `0.910x` | `144` |
| NIAH | `0.077s` | `0.245s` | `0.313x` | `1.410s` | `1.576s` | `0.895x` | `96` |

## Interpretation

Cachet's SGLang path is functionally working on the target hardware: request
metadata reaches HiCache, generated handoff page keys match live runtime keys,
external cache hits are validated, and quality gates pass. The current prepared
prompts are short enough that Cachet's handoff overhead exceeds the saved
prefill time.

## Provenance

`success_run.json` is a compact, sanitized snapshot of the successful live
benchmark, including import probes, handoff generation, coverage, measurements,
comparisons, and cache-hit validations.
