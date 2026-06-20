# Engine Probe Bundle

This standalone Databricks Asset Bundle template runs one native vLLM or SGLang
engine-probe job on a single AWS g5 node. It is for release evidence after an
engine adapter has produced a `document_kv.engine_adapter_request.v1` handoff and
workspace-specific native probe factory.

Run bundle commands from this folder because `databricks.yml` is the bundle root.

```bash
databricks bundle validate \
  --var runner_python_file=dbfs:/benchmarks/run_engine_probe.py \
  --var handoff_json=/Volumes/catalog/schema/volume/probes/vllm-handoff.json \
  --var probe_factory=my_engine_adapter.probes:build_probe \
  --var probe_output_json=/Volumes/catalog/schema/volume/probes/vllm-engine-probe.json \
  --var payload_uri=/Volumes/catalog/schema/volume/probes/vllm-payload.kv \
  --var expected_backend=vllm \
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --var single_user_name=user@example.com
```

The template intentionally omits debug-only `--allow-non-native-probe` and
caller-supplied `--engine-version`. Release evidence expects each native factory
to report the real serving-engine version through `EngineKVProbeFactoryResult`.
Use the Python `databricks_engine_probe_job` helper instead when the job should
also write the optional `document_kv.engine_kv_connector_actions.v1` descriptor
sidecar.

For non-bundle release operations, the package CLI can emit one `runs/submit`
payload with both required native backend probes:

```bash
python -m document_kv_cache.databricks_engine_probe_job \
  --backend-config-json engine-probe-targets.json \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json databricks-engine-probes-submit.json
```

The target JSON must contain exactly one `vllm` probe and one `sglang` probe in
`--release-safe` mode.
`document_kv_cache.benchmark_plan --engine-probe-targets-output-json ...` can
generate that target JSON from the same planned probe artifacts consumed by
release evidence.
