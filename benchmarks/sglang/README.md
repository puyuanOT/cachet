# SGLang Benchmarks

SGLang currently has Cachet correctness/cache-hit evidence through HiCache.
The successful short-prompt runs do not show a latency speedup.

## Experimental Setup

| Result | Model | Hardware | Method | Baseline arm | Cache arm | Dataset scope | Repeats / measurements | Prompt-token scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`qwen3-4b-g6-l4-vanilla-kv-prepared/`](qwen3-4b-g6-l4-vanilla-kv-prepared/) | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV via HiCache | `baseline_prefill` | `document_kv_cache` | Biography, HotpotQA, MusiQue, NIAH | 2 repeats per arm/dataset; 16 measurements | report-row prompt mean not recorded; cache-validation prompt tokens 120-189 |
| [`qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/`](qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/) | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV via HiCache | `baseline_prefill` | `document_kv_cache` | Synthetic NIAH | 2 repeats per arm; 4 measurements | report-row mean prompt tokens 92; cache-validation prompt tokens 205 |

## Main Latency Results

| Result | Baseline p50 TTFT | Cachet p50 TTFT | TTFT speedup | Baseline p50 TTC | Cachet p50 TTC | TTC speedup | Cache-hit validation | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Prepared V1 suite | 0.077-0.197 s | 0.204-0.257 s | 0.31x-0.97x | 0.727-1.416 s | 0.753-1.585 s | 0.89x-0.97x | 8/8 passed; 96-144 cached tokens | Correctness/cache-hit benchmark; no speedup |
| Synthetic NIAH | 0.303 s | 0.346 s | 0.875x | 0.556 s | 0.601 s | 0.926x | 2/2 passed; 175 cached tokens | Tiny-prompt cache-hit check; no speedup |

## Quality Results

| Result | Answer-found baseline/cache | Answer-found delta | Exact-match baseline/cache | Exact-match delta | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Prepared V1 suite | 1.0 / 1.0 | 0.0 | 0.0 / 0.0 | 0.0 | Four datasets |
| Synthetic NIAH | 1.0 / 1.0 | 0.0 | 1.0 / 1.0 | 0.0 | One synthetic prompt |

## Memory / Footprint

| Metric | Prepared V1 suite | Synthetic NIAH | Notes |
| --- | --- | --- | --- |
| Validated cached tokens | 96-144 | 175 | Cache-hit validation only |
| Peak GPU memory | not measured | not measured | Absent from SGLang serving evidence |
| CPU RSS | not measured | not measured | Absent from SGLang serving evidence |
| Cache-resident footprint | not measured | not measured | Absent from SGLang serving evidence |
| Storage throughput | not measured here | not measured here | See [`../storage/`](../storage/) for reader throughput only |

## Coverage

| Model | Hardware | Method | Status |
| --- | --- | --- | --- |
| Qwen3 4B Instruct | AWS g6/L4 | Vanilla external KV via HiCache | Correctness/cache-hit benchmark; no speedup |
| Qwen3 4B Instruct | AWS g6/L4 | KV Packet | not benchmarked yet |
| Other models | Any hardware | Any method | not benchmarked yet |

## Historical Runs

Failed and superseded SGLang smoke attempts are preserved under
[`../../docs/release-ops/benchmark-archive/sglang-smoke/`](../../docs/release-ops/benchmark-archive/sglang-smoke/).
They are useful for maintainers, but they are not public benchmark results.
