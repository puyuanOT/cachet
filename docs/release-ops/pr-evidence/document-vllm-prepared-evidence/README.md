# Prepared vLLM Evidence Gap

This folder records traceability for PR #293, which documents the target g6/L4
prepared vLLM smoke result and the strict release gap it exposed.

## Prepared vLLM Smoke Evidence

- Run id: `49121009370691`
- Benchmark id: `cachet_vllm_prepared_long_20260621_121543_vllm_prepared_long`
- Databricks task purpose: `document-kv-vllm-smoke`
- Cluster: single-node `g6.8xlarge`, Databricks Runtime
  `15.4.x-gpu-ml-scala2.12`, `SINGLE_USER`
- UC artifact root:
  `/Volumes/datascience_qa/kv_cache_restaurant_cls/kv_cache_storage_benchmark/cachet_vllm_prepared_long_20260621_121543/vllm_prepared_long`

| Artifact | Local evidence copy | SHA-256 |
| --- | --- | --- |
| Metadata | `databricks-runs/cachet_vllm_prepared_long_20260621_121543/vllm-prepared-long-metadata.json` | `1d9ecdef2c6d49c05206c740a22da435fd323c7604fdefa229280c82d1fa6a30` |
| Prompt-token budget | `databricks-runs/cachet_vllm_prepared_long_20260621_121543/vllm-prepared-long-prompt-token-budget.json` | `a30142892394ce44dffc4562ca6dcf2a0860e42dfde696a53c8620d69589ef1c` |
| V1 benchmark report | `databricks-runs/cachet_vllm_prepared_long_20260621_121543/vllm-prepared-long-v1-benchmark.json` | `a32d3fdfbb1fd718207c6af2c2bd1e85c537fb57be85ca940a503ba34301c1fc` |
| Databricks run status | `databricks-runs/cachet_vllm_prepared_long_20260621_121543/vllm-prepared-long-run-status.json` | `3dba5a6ea65a253676676e7748d9406ae9d325e6ef85f318bc5691089808123d` |

Observed result: the prepared vLLM smoke run completed successfully and produced
`v1_evidence.ok=true`. Prompt-token preflight reported no over-budget rows:
Biography used 19,262 prompt tokens, HotpotQA 28,907, MusiQue 27,702, and NIAH
28,867, all with `max_tokens=100` under `max_model_len=32768`.

Strict release gap: this is plain vLLM OpenAI-compatible smoke evidence. It is
not strict cache-release evidence because the cache arm still sends the full
logical prompt (`prompt_text_mode=logical`) instead of only a runtime suffix with
`runtime_prompt_tokens < logical_prompt_tokens`.

## UC Storage Evidence

- Run id: `948365719597221`
- Benchmark id: `cachet_readiness_20260621_095026_storage`
- Databricks task purpose: `document-kv-storage-benchmark`
- Cluster: single-node `g6.8xlarge`, Databricks Runtime
  `15.4.x-gpu-ml-scala2.12`, `SINGLE_USER`
- UC storage benchmark output:
  `/Volumes/datascience_qa/kv_cache_restaurant_cls/kv_cache_storage_benchmark/cachet_readiness_20260621_095026/storage/storage-benchmark.json`
- UC reader root:
  `/Volumes/datascience_qa/kv_cache_restaurant_cls/kv_cache_storage_benchmark/cachet_readiness_20260621_095026/storage/uc-root`

| Artifact | Local evidence copy | SHA-256 |
| --- | --- | --- |
| Storage benchmark report | `databricks-runs/cachet_readiness_20260621_095026/storage-result.json` | `0cbb80feb6ef8699bb900efdaf59efbd7c4d4d725211077315e489d3ca653549` |
| Databricks run status | `databricks-runs/cachet_readiness_20260621_095026/storage-benchmark-run-status.json` | `25c0e314684452aa3a34968ace71e7d5df75baec48cf54241207984d37c603d5` |

Observed result: release storage evidence was `ok=true`; Memory, Disk, and Unity
Catalog readers completed with zero errors against a real UC Volume. The
remaining release action is to include these artifacts in the strict bundle
alongside valid benchmark, engine-probe, connector-action, launch-config,
governance, hygiene, PR-evidence, and native-probe-factory sidecars.
