# Current Benchmark Results

This page is the fast benchmark appendix for Cachet. It uses only committed
sanitized evidence and marks missing measurements explicitly.

## Experimental Setup

| Result | Engine | Model | Hardware | Method | Baseline arm | Cache arm | Dataset scope | Repeats / measurements | Prompt-token scope | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vLLM primary | vLLM | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV | `baseline_prefill` | `document_kv_cache` | Biography, HotpotQA, MusiQue, NIAH | 3 repeats per arm/dataset; 24 measurements | mean prompt tokens 15,491-23,231 | [`../vllm/qwen3-4b-g6-l4-vanilla-kv/v1_benchmark.json`](../vllm/qwen3-4b-g6-l4-vanilla-kv/v1_benchmark.json) |
| vLLM compatibility | vLLM | `qwen3:4b-instruct` | AWS g5/A10G, `g5.8xlarge` | Vanilla external KV | `baseline_prefill` | `document_kv_cache` | Biography, HotpotQA, MusiQue, NIAH | 3 repeats per arm/dataset; 24 measurements | mean prompt tokens 15,491-23,231 | [`../vllm/qwen3-4b-g5-a10g-vanilla-kv/v1_benchmark.json`](../vllm/qwen3-4b-g5-a10g-vanilla-kv/v1_benchmark.json) |
| SGLang prepared | SGLang | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV via HiCache | `baseline_prefill` | `document_kv_cache` | Biography, HotpotQA, MusiQue, NIAH | 2 repeats per arm/dataset; 16 measurements | report-row prompt mean not recorded; cache-validation prompt tokens 120-189 | [`../sglang/qwen3-4b-g6-l4-vanilla-kv-prepared/success_run.json`](../sglang/qwen3-4b-g6-l4-vanilla-kv-prepared/success_run.json) |
| SGLang synthetic | SGLang | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV via HiCache | `baseline_prefill` | `document_kv_cache` | Synthetic NIAH | 2 repeats per arm; 4 measurements | report-row mean prompt tokens 92; cache-validation prompt tokens 205 | [`../sglang/qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/success_run.json`](../sglang/qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/success_run.json) |
| Storage readers | Cachet readers | n/a | AWS g6/L4, `g6.8xlarge` | Memory, disk, Unity Catalog readers | n/a | n/a | 256 MiB read workload per reader | 256 reads; parallelism 8 | n/a | [`../storage/g6-l4-reader-throughput/storage_benchmark.json`](../storage/g6-l4-reader-throughput/storage_benchmark.json) |
| Native connector probes | vLLM and SGLang | `qwen3:4b-instruct` layout probe | AWS g6/L4, `g6.8xlarge` | Vanilla external KV handoff | n/a | provider-backed native connector | 48-token fixture | one probe per engine | copied tokens 48 | [`../native-engine/g6-l4-vllm-sglang-vanilla-kv/`](../native-engine/g6-l4-vllm-sglang-vanilla-kv/) |

## Main Latency Results

Speedup is `baseline p50 / Cachet p50`. Values below `1.0x` mean Cachet was
slower than the baseline in that run.

| Result | Baseline p50 TTFT | Cachet p50 TTFT | TTFT speedup | Baseline p50 time-to-completion | Cachet p50 time-to-completion | TTC speedup | Repeats | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM primary g6/L4 | 4.841-9.188 s | 0.919-1.583 s | 5.27x-6.97x | 9.205-14.126 s | 5.285-6.451 s | 1.74x-2.25x | 3 | Speedup benchmark |
| vLLM compatibility g5/A10G | 4.460-8.116 s | 0.950-1.632 s | 4.66x-6.04x | 6.901-10.816 s | 3.388-4.310 s | 2.04x-2.67x | 3 | Compatibility benchmark |
| SGLang prepared g6/L4 | 0.077-0.197 s | 0.204-0.257 s | 0.31x-0.97x | 0.727-1.416 s | 0.753-1.585 s | 0.89x-0.97x | 2 | Correctness/cache-hit benchmark; no speedup |
| SGLang synthetic NIAH | 0.303 s | 0.346 s | 0.875x | 0.556 s | 0.601 s | 0.926x | 2 | Tiny-prompt cache-hit check; no speedup |
| Storage readers | n/a | n/a | not a serving latency benchmark | n/a | n/a | not a serving latency benchmark | 256 reads | See footprint table |
| Native connector probes | not measured | not measured | not measured | not measured | not measured | not measured | one probe per engine | Integration evidence only |

## Quality Results

`answer_found_rate` is the current quality gate: whether the expected answer
appears in the model output. `exact_match_rate` is a stricter literal match
reported in the raw evidence. In the current vLLM and SGLang prepared evidence,
quality delta `0.0` means answer-found delta `0.0`; the raw exact-match rate is
also shown below.

| Result | Baseline answer-found | Cachet answer-found | Answer-found delta | Baseline exact-match | Cachet exact-match | Exact-match delta | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM primary g6/L4 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | Four datasets |
| vLLM compatibility g5/A10G | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | Four datasets |
| SGLang prepared g6/L4 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | Four datasets; 8/8 cache-hit validations |
| SGLang synthetic NIAH | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | One synthetic prompt; 2/2 cache-hit validations |
| Storage readers | n/a | n/a | n/a | n/a | n/a | n/a | Not a generation benchmark |
| Native connector probes | n/a | n/a | n/a | n/a | n/a | n/a | Not a generation benchmark |

## Memory / Footprint

Storage throughput is not memory consumption. Serving peak GPU memory, CPU RSS,
and cache-resident footprint are not measured in the current serving benchmark
artifacts.

| Evidence | Bytes | Throughput | p50 latency | Copied tokens | Bytes per token | Estimated GPU bytes | Peak GPU memory | CPU RSS | Cache footprint | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Storage memory reader | 268,435,456 total bytes | 6531.4 MiB/s | 0.847 ms | n/a | n/a | n/a | not measured | not measured | not measured | Reader throughput only |
| Storage disk reader | 268,435,456 total bytes | 6214.4 MiB/s | 1.130 ms | n/a | n/a | n/a | not measured | not measured | not measured | Reader throughput only |
| Storage Unity Catalog reader | 268,435,456 total bytes | 1148.0 MiB/s | 5.332 ms | n/a | n/a | n/a | not measured | not measured | not measured | Reader throughput only |
| Native vLLM probe | 3,538,944 copied bytes | n/a | n/a | 48 | 73,728 | not measured | not measured | not measured | not measured | Engine-owned KV block-manager path |
| Native SGLang probe | 3,538,944 copied bytes | n/a | n/a | 48 | 73,728 | not measured | not measured | not measured | not measured | HiCache-backed path |
| vLLM serving benchmark | not measured | n/a | n/a | not measured | not measured | not measured | not measured | not measured | not measured | Latency and quality only |
| SGLang serving benchmark | not measured | n/a | n/a | cached-token validations 96-175 | not measured | not measured | not measured | not measured | not measured | Latency, quality, and cache-hit validation only |

## Coverage Matrix

| Engine | Model | Hardware | Method | Status |
| --- | --- | --- | --- | --- |
| vLLM | Qwen3 4B Instruct | AWS g6/L4 | Vanilla external KV | Latency speedup benchmark |
| vLLM | Qwen3 4B Instruct | AWS g5/A10G | Vanilla external KV | Compatibility benchmark |
| SGLang | Qwen3 4B Instruct | AWS g6/L4 | Vanilla external KV via HiCache | Correctness/cache-hit benchmark; no speedup |
| vLLM / SGLang | Qwen3 4B layout fixture | AWS g6/L4 | Native connector handoff | Integration evidence |
| Any engine | Any model | Any hardware | KV Packet | not benchmarked yet |
| Any engine | Any model | Any hardware | Learned or adapter-trained KV methods | not benchmarked yet |
| vLLM / SGLang | Other models | Any hardware | Any method | not benchmarked yet |

## Limitations

| Limitation | Current state |
| --- | --- |
| Public model coverage | One public model: Qwen3 4B Instruct |
| Method coverage | One benchmarked method: vanilla external KV |
| Repeat count | vLLM uses 3 repeats per arm/dataset; SGLang uses 2 repeats |
| Peak memory | GPU peak memory, CPU RSS, and serving cache footprint are not measured |
| SGLang | Integration/correctness only; current runs do not show speedup |
| Storage | Reader throughput only; not serving latency or memory consumption |

## Provenance

Use the result folders for sanitized JSON evidence:

- [`../vllm/qwen3-4b-g6-l4-vanilla-kv/`](../vllm/qwen3-4b-g6-l4-vanilla-kv/)
- [`../vllm/qwen3-4b-g5-a10g-vanilla-kv/`](../vllm/qwen3-4b-g5-a10g-vanilla-kv/)
- [`../sglang/qwen3-4b-g6-l4-vanilla-kv-prepared/`](../sglang/qwen3-4b-g6-l4-vanilla-kv-prepared/)
- [`../sglang/qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/`](../sglang/qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/)
- [`../storage/g6-l4-reader-throughput/`](../storage/g6-l4-reader-throughput/)
- [`../native-engine/g6-l4-vllm-sglang-vanilla-kv/`](../native-engine/g6-l4-vllm-sglang-vanilla-kv/)

Use [`../databricks/CURRENT.md`](../databricks/CURRENT.md) only when you need
QA run IDs or release-source mirrors.
