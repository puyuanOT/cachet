# Current Benchmark Results

This is the quick answer for users evaluating Cachet. It compares Cachet with
the no-cache prefill baseline and shows which engines, methods, hardware, and
models are covered.

## Performance Summary

| Engine | Model | Hardware | Method | Dataset Scope | Cachet Result |
| --- | --- | --- | --- | --- | --- |
| vLLM | Qwen3 4B Instruct | AWS g6/L4, `g6.8xlarge` | Vanilla external KV | Biography, HotpotQA, MusiQue, NIAH | 5.27x-6.97x TTFT speedup; 1.74x-2.25x time-to-completion speedup; quality delta `0.0` |
| vLLM | Qwen3 4B Instruct | AWS g5/A10G, `g5.8xlarge` | Vanilla external KV | Biography, HotpotQA, MusiQue, NIAH | 4.66x-6.04x TTFT speedup; 2.04x-2.67x time-to-completion speedup; quality delta `0.0` |
| SGLang | Qwen3 4B Instruct | AWS g6/L4, `g6.8xlarge` | Vanilla external KV via HiCache | Prepared V1 Biography, HotpotQA, MusiQue, NIAH | 16 measurements; 8/8 Cachet-backed cache hits; quality delta `0.0`; no speedup on short prompts |
| SGLang | Qwen3 4B Instruct | AWS g6/L4, `g6.8xlarge` | Vanilla external KV via HiCache | Synthetic NIAH live prompt | Two Cachet-backed cache repeats with 175 cached tokens; quality delta `0.0`; tiny-prompt scoped evidence |

The strongest current performance claim is vLLM on g6/L4. SGLang currently
shows that Cachet can hydrate external KV through the live serving path with
correct answers, but the measured short-prompt runs are slower than the
baseline.

## Stable Result Folders

| Result Folder | What To Use It For |
| --- | --- |
| [`../vllm/qwen3-4b-g6-l4-vanilla-kv/`](../vllm/qwen3-4b-g6-l4-vanilla-kv/) | Primary vLLM performance result on the target g6/L4 hardware |
| [`../vllm/qwen3-4b-g5-a10g-vanilla-kv/`](../vllm/qwen3-4b-g5-a10g-vanilla-kv/) | g5/A10G compatibility performance result |
| [`../sglang/qwen3-4b-g6-l4-vanilla-kv-prepared/`](../sglang/qwen3-4b-g6-l4-vanilla-kv-prepared/) | SGLang prepared live V1 correctness and cache-hit benchmark |
| [`../sglang/qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/`](../sglang/qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/) | Small SGLang synthetic live benchmark used to validate repeated live cache hits |
| [`../storage/g6-l4-reader-throughput/`](../storage/g6-l4-reader-throughput/) | Memory, disk, and Unity Catalog reader throughput and latency |
| [`../native-engine/g6-l4-vllm-sglang-vanilla-kv/`](../native-engine/g6-l4-vllm-sglang-vanilla-kv/) | vLLM and SGLang native connector integration evidence |

## Supported Methods

| Method | vLLM | SGLang | Notes |
| --- | --- | --- | --- |
| Vanilla external KV cache | Benchmarked with speedups | Live cache-hit and quality gates pass; no speedup yet | Current Cachet V1 method |
| KV Packet | Not yet benchmarked | Not yet benchmarked | Planned extension |
| Adapter-trained or learned KV methods | Not yet benchmarked | Not yet benchmarked | API has placeholders, but no public result |

## Provenance

The report folders include sanitized JSON evidence beside the summaries. Use
[`../databricks/CURRENT.md`](../databricks/CURRENT.md) only when you need QA run
IDs or release-source mirrors. Use
[`../../docs/release-ops/benchmark-archive/`](../../docs/release-ops/benchmark-archive/)
for historical failed or superseded SGLang readiness runs.
