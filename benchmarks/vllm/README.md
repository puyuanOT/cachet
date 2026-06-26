# vLLM Benchmarks

These folders compare vLLM no-cache prefill with Cachet vanilla external KV.
The g6/L4 run is the primary performance result; the g5/A10G run is
compatibility evidence.

## Experimental Setup

| Result | Model | Hardware | Method | Baseline arm | Cache arm | Dataset scope | Repeats / measurements | Prompt-token mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`qwen3-4b-g6-l4-vanilla-kv/`](qwen3-4b-g6-l4-vanilla-kv/) | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV | `baseline_prefill` | `document_kv_cache` | Biography, HotpotQA, MusiQue, NIAH | 3 repeats per arm/dataset; 24 measurements | 15,491-23,231 |
| [`qwen3-4b-g5-a10g-vanilla-kv/`](qwen3-4b-g5-a10g-vanilla-kv/) | `qwen3:4b-instruct` | AWS g5/A10G, `g5.8xlarge` | Vanilla external KV | `baseline_prefill` | `document_kv_cache` | Biography, HotpotQA, MusiQue, NIAH | 3 repeats per arm/dataset; 24 measurements | 15,491-23,231 |

## Main Latency Results

| Result | Baseline p50 TTFT | Cachet p50 TTFT | TTFT speedup | Baseline p50 TTC | Cachet p50 TTC | TTC speedup | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| g6/L4 primary | 4.841-9.188 s | 0.919-1.583 s | 5.27x-6.97x | 9.205-14.126 s | 5.285-6.451 s | 1.74x-2.25x | Speedup benchmark |
| g5/A10G compatibility | 4.460-8.116 s | 0.950-1.632 s | 4.66x-6.04x | 6.901-10.816 s | 3.388-4.310 s | 2.04x-2.67x | Compatibility benchmark |

## Quality Results

| Result | Answer-found baseline/cache | Answer-found delta | Exact-match baseline/cache | Exact-match delta | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| g6/L4 primary | 1.0 / 1.0 | 0.0 | 0.0 / 0.0 | 0.0 | `answer_found_rate` is the current quality gate |
| g5/A10G compatibility | 1.0 / 1.0 | 0.0 | 0.0 / 0.0 | 0.0 | Raw exact-match is reported separately |

## Memory / Footprint

| Metric | g6/L4 | g5/A10G | Notes |
| --- | --- | --- | --- |
| Peak GPU memory | not measured | not measured | Absent from serving benchmark JSON |
| CPU RSS | not measured | not measured | Absent from serving benchmark JSON |
| Cache-resident footprint | not measured | not measured | Absent from serving benchmark JSON |
| Storage throughput | not measured here | not measured here | See [`../storage/`](../storage/) for reader throughput only |

## Coverage

| Model | Hardware | Method | Status |
| --- | --- | --- | --- |
| Qwen3 4B Instruct | AWS g6/L4 | Vanilla external KV | Speedup benchmark |
| Qwen3 4B Instruct | AWS g5/A10G | Vanilla external KV | Compatibility benchmark |
| Qwen3 4B Instruct | Any hardware | KV Packet | not benchmarked yet |
| Other models | Any hardware | Any method | not benchmarked yet |

Databricks run IDs and sanitized run-status mirrors live under
[`../databricks/`](../databricks/) for audits.
