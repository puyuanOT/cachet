# Existing Benchmark Evidence

These folders preserve committed benchmark and integration evidence that does
not match the fixed main-table configuration in [benchmark root](../../).
They remain useful for audits, release evidence, and implementation history.

| Result folder | Evidence type | Key result | Main-table mismatch |
| --- | --- | --- | --- |
| [`vllm-qwen3-4b-g6-l4-vanilla-kv/`](vllm-qwen3-4b-g6-l4-vanilla-kv/) | vLLM latency and quality | 5.27x-6.97x TTFT speedup; answer-found delta `0.0` | g6/L4, 3 repeats, prompt-token means 15,491-23,231, 100-token completions |
| [`vllm-qwen3-4b-g5-a10g-vanilla-kv/`](vllm-qwen3-4b-g5-a10g-vanilla-kv/) | vLLM compatibility latency and quality | 4.66x-6.04x TTFT speedup; answer-found delta `0.0` | 3 repeats, prompt-token means 15,491-23,231, 100-token completions |
| [`sglang-qwen3-4b-g6-l4-vanilla-kv-prepared/`](sglang-qwen3-4b-g6-l4-vanilla-kv-prepared/) | SGLang correctness/cache-hit evidence | 8/8 cache-hit validations; no speedup on short prompts | SGLang, g6/L4, short prompts, 2 repeats |
| [`sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/`](sglang-qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/) | SGLang synthetic cache-hit check | 2/2 repeated cache-hit validations; no speedup | One tiny synthetic prompt |
| [`storage-g6-l4-reader-throughput/`](storage-g6-l4-reader-throughput/) | Storage-reader throughput | Memory 6531.4 MiB/s, disk 6214.4 MiB/s, Unity Catalog 1148.0 MiB/s | Reader throughput only, not serving TTFT/TTC |
| [`native-engine-g6-l4-vllm-sglang-vanilla-kv/`](native-engine-g6-l4-vllm-sglang-vanilla-kv/) | Native connector integration | vLLM and SGLang probes copied 48 tokens / 3,538,944 bytes | Integration and copied-byte evidence only |

The Databricks mirror records remain at [`../../databricks/`](../../databricks/)
because release-evidence JSON uses those paths.
