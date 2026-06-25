# Current Databricks Benchmark Mirrors

This page records the QA run provenance behind the public benchmark reports.
It is intentionally secondary to [`../current/`](../current/).

| Public Result | Mirror | QA Run | Hardware | Result |
| --- | --- | --- | --- | --- |
| vLLM Qwen3 4B, vanilla KV | [`vllm-qwen3-4b-g6-l4-vanilla-kv/`](vllm-qwen3-4b-g6-l4-vanilla-kv/) | `872615985402004` | `aws-g6-l4` / `g6.8xlarge` | 24 measurements; 5.27x-6.97x TTFT speedup; quality delta `0.0` |
| vLLM Qwen3 4B, vanilla KV compatibility | [`vllm-qwen3-4b-g5-a10g-vanilla-kv/`](vllm-qwen3-4b-g5-a10g-vanilla-kv/) | `566743786103032` | `aws-g5-a10g` / `g5.8xlarge` | 24 measurements; 4.66x-6.04x TTFT speedup; quality delta `0.0` |
| Storage readers | [`storage-g6-l4-reader-throughput/`](storage-g6-l4-reader-throughput/) | `948365719597221` | `aws-g6-l4` / `g6.8xlarge` | Memory, disk, and Unity Catalog readers; zero reader errors |
| Native vLLM/SGLang connector probes | [`native-engine-g6-l4-vllm-sglang-vanilla-kv/`](native-engine-g6-l4-vllm-sglang-vanilla-kv/) | `934698284395881` | `aws-g6-l4` / `g6.8xlarge` | Provider-backed vLLM and SGLang probes succeeded |
| SGLang prepared live V1 | [`../sglang/qwen3-4b-g6-l4-vanilla-kv-prepared/`](../sglang/qwen3-4b-g6-l4-vanilla-kv-prepared/) | `48413356233422` | `aws-g6-l4` / `g6.8xlarge` | 16 measurements; 8/8 Cachet-backed cache-hit validations; no speedup |
| SGLang synthetic NIAH | [`../sglang/qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/`](../sglang/qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/) | `238535418152934` | `aws-g6-l4` / `g6.8xlarge` | Two Cachet-backed cache repeats with 175 cached tokens; no speedup |

The strict publication target remains AWS g6/L4. g5/A10G is compatibility
evidence only.
