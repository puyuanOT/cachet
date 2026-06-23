# AWS g6/L4 Native Engine Probes

This folder records the current provider-backed native connector probes for
vLLM and SGLang on the strict AWS g6/L4 Databricks target. These probes verify
the native engine handoff path; they are not benchmark latency measurements.

| Field | Value |
| --- | --- |
| Databricks run | `934698284395881` |
| Run name | `document-kv-engine-probe` |
| Hardware target | `aws-g6-l4` |
| Node type | `g6.8xlarge` |
| Task count | 2 |
| Result | Both tasks terminated `SUCCESS` |

## Results

| Backend | Engine version | Payload mode | Copied tokens | Copied bytes | Provider factory |
| --- | --- | --- | ---: | ---: | --- |
| vllm | 0.23.0 | merged | 48 | 3538944 | `vllm_kv_injection.vllm_native_provider:build_document_kv_provider` |
| sglang | 0.5.10.post1 | merged | 48 | 3538944 | `sglang_kv_injection.sglang_dynamic_backend:build_document_kv_hicache_provider` |

## Artifacts

- [`databricks_run_status.json`](databricks_run_status.json) contains the
  terminal successful Databricks run-status summary.
- [`vllm_engine_probe.json`](vllm_engine_probe.json) and
  [`sglang_engine_probe.json`](sglang_engine_probe.json) contain the native
  connector probe records.
- [`vllm_connector_actions.json`](vllm_connector_actions.json) and
  [`sglang_connector_actions.json`](sglang_connector_actions.json) contain the
  connector action descriptors used by strict release validation.
