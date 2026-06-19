# Databricks Templates

This folder contains Databricks Asset Bundle templates packaged with the wheel.
It mirrors the repository-level `databricks/` folder so an installed package can
provide the same benchmark and smoke-test scaffolding without a source checkout.

- `databricks.yml` defines the full document KV-cache benchmark job template.
- `engine-probe/` contains the native vLLM/SGLang engine-probe bundle template.
- `storage-benchmark/` contains the storage-reader benchmark bundle template.
- `vllm-smoke/` contains the self-contained Qwen3/vLLM smoke bundle template.

All templates expect caller-supplied variables for workspace paths, wheel URIs,
and the single-user identity used by Unity Catalog enabled clusters.
