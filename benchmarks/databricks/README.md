# Databricks Benchmark Mirrors

This folder is an audit mirror, not the primary benchmark summary. Start with
the [benchmark root](../) for the user-facing performance tables.

The folders here mirror sanitized QA Databricks records for appendix evidence.
They stay at stable paths because release-evidence JSON refers to them:

| Mirror | Public Report | Purpose |
| --- | --- | --- |
| [`vllm-qwen3-4b-g6-l4-vanilla-kv/`](vllm-qwen3-4b-g6-l4-vanilla-kv/) | [`../appendix/existing-results/vllm-qwen3-4b-g6-l4-vanilla-kv/`](../appendix/existing-results/vllm-qwen3-4b-g6-l4-vanilla-kv/) | Existing vLLM g6/L4 benchmark |
| [`vllm-qwen3-4b-g5-a10g-vanilla-kv/`](vllm-qwen3-4b-g5-a10g-vanilla-kv/) | [`../appendix/existing-results/vllm-qwen3-4b-g5-a10g-vanilla-kv/`](../appendix/existing-results/vllm-qwen3-4b-g5-a10g-vanilla-kv/) | Existing vLLM g5/A10G compatibility benchmark |
| [`storage-g6-l4-reader-throughput/`](storage-g6-l4-reader-throughput/) | [`../appendix/existing-results/storage-g6-l4-reader-throughput/`](../appendix/existing-results/storage-g6-l4-reader-throughput/) | Storage-reader benchmark |
| [`native-engine-g6-l4-vllm-sglang-vanilla-kv/`](native-engine-g6-l4-vllm-sglang-vanilla-kv/) | [`../appendix/existing-results/native-engine-g6-l4-vllm-sglang-vanilla-kv/`](../appendix/existing-results/native-engine-g6-l4-vllm-sglang-vanilla-kv/) | vLLM/SGLang native connector evidence |

These mirrors contain compact schema-validated JSON such as
`document_kv.benchmark_run.v1`, `document_kv.databricks_run_status.v1`,
`document_kv.storage_benchmark.v1`, and engine connector probe/action records.

Do not put raw Jobs API responses, credentials, wheels, logs, generated
datasets, or local `databricks-runs/` output here.
