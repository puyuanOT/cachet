# Databricks Benchmark Mirrors

This folder is for audits, not first-time benchmark reading. Start with
[`../current/`](../current/) for the user-facing performance summary.

The folders here mirror sanitized QA Databricks records for the public
benchmark reports:

| Mirror | Public Report | Purpose |
| --- | --- | --- |
| [`vllm-qwen3-4b-g6-l4-vanilla-kv/`](vllm-qwen3-4b-g6-l4-vanilla-kv/) | [`../vllm/qwen3-4b-g6-l4-vanilla-kv/`](../vllm/qwen3-4b-g6-l4-vanilla-kv/) | Primary vLLM g6/L4 benchmark |
| [`vllm-qwen3-4b-g5-a10g-vanilla-kv/`](vllm-qwen3-4b-g5-a10g-vanilla-kv/) | [`../vllm/qwen3-4b-g5-a10g-vanilla-kv/`](../vllm/qwen3-4b-g5-a10g-vanilla-kv/) | vLLM g5/A10G compatibility benchmark |
| [`storage-g6-l4-reader-throughput/`](storage-g6-l4-reader-throughput/) | [`../storage/g6-l4-reader-throughput/`](../storage/g6-l4-reader-throughput/) | Storage-reader benchmark |
| [`native-engine-g6-l4-vllm-sglang-vanilla-kv/`](native-engine-g6-l4-vllm-sglang-vanilla-kv/) | [`../native-engine/g6-l4-vllm-sglang-vanilla-kv/`](../native-engine/g6-l4-vllm-sglang-vanilla-kv/) | vLLM/SGLang native connector evidence |

These mirrors contain compact schema-validated JSON such as
`document_kv.benchmark_run.v1`, `document_kv.databricks_run_status.v1`,
`document_kv.storage_benchmark.v1`, and engine connector probe/action records.

Do not put raw Jobs API responses, credentials, wheels, logs, generated
datasets, or local `databricks-runs/` output here.
