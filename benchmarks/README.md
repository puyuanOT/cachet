# Cachet Benchmarks

This directory is the public benchmark appendix for Cachet. It is organized for
readers who want the speedup, quality, footprint evidence, and coverage gaps
without reading release-audit sidecars first.

Start with [`current/`](current/). It answers what was tested, on what
hardware, how many times, how fast, with what quality result, and what is still
missing.

## Headline Results

| Area | Best current answer | Evidence |
| --- | --- | --- |
| Primary speedup result | vLLM + Qwen3 4B + AWS g6/L4 + vanilla external KV: 5.27x-6.97x TTFT speedup and 1.74x-2.25x time-to-completion speedup | [`current/`](current/) |
| Compatibility result | vLLM + Qwen3 4B + AWS g5/A10G + vanilla external KV: 4.66x-6.04x TTFT speedup | [`vllm/qwen3-4b-g5-a10g-vanilla-kv/`](vllm/qwen3-4b-g5-a10g-vanilla-kv/) |
| SGLang status | Correctness/cache-hit evidence through HiCache; current short-prompt runs show no speedup | [`sglang/`](sglang/) |
| Storage footprint evidence | 256 MiB reader benchmark: memory 6531.4 MiB/s, disk 6214.4 MiB/s, Unity Catalog 1148.0 MiB/s | [`storage/g6-l4-reader-throughput/`](storage/g6-l4-reader-throughput/) |
| Native integration evidence | vLLM and SGLang native probes each copied 48 tokens / 3,538,944 bytes through engine-owned KV paths | [`native-engine/g6-l4-vllm-sglang-vanilla-kv/`](native-engine/g6-l4-vllm-sglang-vanilla-kv/) |
| Missing footprint metrics | Serving peak GPU memory, CPU RSS, and cache-resident footprint are not measured yet | [`current/`](current/) |

## Experimental Setup Summary

| Result family | Engine | Model | Hardware | Method | Baseline arm | Cache arm | Dataset scope | Repeats / measurements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vLLM primary | vLLM | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV | `baseline_prefill` | `document_kv_cache` | Biography, HotpotQA, MusiQue, NIAH | 3 repeats per arm/dataset; 24 measurements |
| vLLM compatibility | vLLM | `qwen3:4b-instruct` | AWS g5/A10G, `g5.8xlarge` | Vanilla external KV | `baseline_prefill` | `document_kv_cache` | Biography, HotpotQA, MusiQue, NIAH | 3 repeats per arm/dataset; 24 measurements |
| SGLang prepared | SGLang | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV via HiCache | `baseline_prefill` | `document_kv_cache` | Biography, HotpotQA, MusiQue, NIAH | 2 repeats per arm/dataset; 16 measurements |
| SGLang synthetic | SGLang | `qwen3:4b-instruct` | AWS g6/L4, `g6.8xlarge` | Vanilla external KV via HiCache | `baseline_prefill` | `document_kv_cache` | Synthetic NIAH | 2 repeats per arm; 4 measurements |
| Storage readers | Cachet readers | n/a | AWS g6/L4, `g6.8xlarge` | Memory, disk, Unity Catalog | n/a | n/a | 256 MiB read workload | 256 reads per reader |
| Native probes | vLLM and SGLang | `qwen3:4b-instruct` fixture | AWS g6/L4, `g6.8xlarge` | Vanilla external KV handoff | n/a | provider-backed connector | 48-token fixture | one probe per engine |

## Main Latency Summary

| Result | Baseline p50 TTFT | Cachet p50 TTFT | TTFT speedup | Baseline p50 TTC | Cachet p50 TTC | TTC speedup | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM g6/L4 | 4.841-9.188 s | 0.919-1.583 s | 5.27x-6.97x | 9.205-14.126 s | 5.285-6.451 s | 1.74x-2.25x | Speedup benchmark |
| vLLM g5/A10G | 4.460-8.116 s | 0.950-1.632 s | 4.66x-6.04x | 6.901-10.816 s | 3.388-4.310 s | 2.04x-2.67x | Compatibility benchmark |
| SGLang prepared g6/L4 | 0.077-0.197 s | 0.204-0.257 s | 0.31x-0.97x | 0.727-1.416 s | 0.753-1.585 s | 0.89x-0.97x | Correctness/cache-hit evidence; no speedup |
| SGLang synthetic NIAH | 0.303 s | 0.346 s | 0.875x | 0.556 s | 0.601 s | 0.926x | Tiny-prompt cache-hit evidence; no speedup |
| Storage readers | n/a | n/a | not a serving benchmark | n/a | n/a | not a serving benchmark | Reader throughput only |
| Native probes | not measured | not measured | not measured | not measured | not measured | not measured | Integration evidence only |

## Quality And Footprint

| Topic | Current evidence | Missing evidence |
| --- | --- | --- |
| Quality metric | `answer_found_rate` is 1.0 for baseline and Cachet in current vLLM and SGLang successful runs; answer-found delta is 0.0 | More human or task-specific quality metrics are not benchmarked here |
| Exact match | vLLM and SGLang prepared raw exact-match rates are 0.0; SGLang synthetic exact-match is 1.0 | Exact-match is not the primary current quality gate |
| Storage bytes | Each reader processed 268,435,456 bytes with zero reader errors | This is not serving memory consumption |
| Native connector footprint | vLLM and SGLang probes copied 3,538,944 bytes for 48 tokens; layout reports 73,728 bytes/token | Estimated GPU bytes field, serving peak GPU memory, CPU RSS, and cache footprint are not measured |

## Coverage Matrix

| Engine / area | Model | Hardware | Method | Status |
| --- | --- | --- | --- | --- |
| vLLM | Qwen3 4B Instruct | AWS g6/L4 | Vanilla external KV | Latency speedup benchmark |
| vLLM | Qwen3 4B Instruct | AWS g5/A10G | Vanilla external KV | Compatibility benchmark |
| SGLang | Qwen3 4B Instruct | AWS g6/L4 | Vanilla external KV via HiCache | Correctness/cache-hit benchmark; no speedup |
| Native connectors | Qwen3 4B fixture | AWS g6/L4 | Vanilla external KV handoff | Integration evidence |
| Storage readers | n/a | AWS g6/L4 | Memory, disk, Unity Catalog | Reader throughput benchmark |
| vLLM / SGLang | Any model | Any hardware | KV Packet | not benchmarked yet |
| vLLM / SGLang | Any model | Any hardware | Learned or adapter-trained KV methods | not benchmarked yet |
| vLLM / SGLang | Other public models | Any hardware | Any method | not benchmarked yet |

## Directory Guide

| Folder | What to read it for |
| --- | --- |
| [`current/`](current/) | Concise appendix with setup, latency, quality, footprint, coverage, limitations, and provenance |
| [`vllm/`](vllm/) | vLLM no-cache prefill versus Cachet latency and quality tables |
| [`sglang/`](sglang/) | SGLang correctness/cache-hit status and no-speedup evidence |
| [`storage/`](storage/) | Storage-reader throughput and latency; not memory consumption |
| [`native-engine/`](native-engine/) | vLLM/SGLang native connector copied-byte evidence |
| [`databricks/`](databricks/) | Sanitized QA run-status mirrors for audits |
| [`_template/`](_template/) | Required table shape for future benchmark folders |

## Limitations

| Limitation | Current state |
| --- | --- |
| Public model coverage | Qwen3 4B Instruct only |
| Public method coverage | Vanilla external KV only |
| Repeat counts | Small: 3 repeats for vLLM, 2 repeats for SGLang |
| Memory | Serving peak GPU memory, CPU RSS, and cache footprint are not measured |
| SGLang | Integration/correctness evidence only; current runs do not show latency speedup |
| Storage | Reader throughput only; not serving latency or memory consumption |

Historical failed SGLang smoke attempts and superseded readiness artifacts live
under [`docs/release-ops/benchmark-archive/`](../docs/release-ops/benchmark-archive/)
instead of the public benchmark path.
