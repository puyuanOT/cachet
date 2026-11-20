# Databricks Templates

This folder contains Databricks Asset Bundle templates packaged with the wheel.
It mirrors the repository-level `databricks/` folder so an installed package can
provide the same benchmark and smoke-test scaffolding without a source checkout.

- `databricks.yml` defines the full document KV-cache benchmark job template.
- `engine-probe/` contains the native vLLM/SGLang engine-probe bundle template.
- `storage-benchmark/` contains the storage-reader benchmark bundle template.
- `vllm-smoke/` contains the self-contained Qwen3/vLLM smoke bundle template.

All templates expect caller-supplied variables for workspace paths, wheel URIs,
and the single-user identity used by Unity Catalog enabled clusters. The main
benchmark bundle also accepts non-secret `transformers_*` variables for
`CACHET_TRANSFORMERS_*` generator runtime settings.
Use the `cachet-templates` CLI from an installed wheel to inspect or
extract these package-data files before running Databricks Asset Bundle
commands:

```bash
cachet-templates list --prefix databricks
cachet-templates extract \
  --prefix databricks \
  --output-dir ./cachet-templates
```

After extraction, run `databricks bundle validate` or deploy commands from the
extracted bundle root, such as `cachet-templates/databricks/`,
`cachet-templates/databricks/storage-benchmark/`,
`cachet-templates/databricks/engine-probe/`, or
`cachet-templates/databricks/vllm-smoke/`.
