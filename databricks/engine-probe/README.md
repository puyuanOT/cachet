# Engine Probe Bundle

This standalone Databricks Asset Bundle template runs one native vLLM or SGLang
engine-probe job on a single AWS g6/L4 node. It is useful for a single native
probe after an engine adapter has produced a `document_kv.engine_adapter_request.v1`
handoff and either a workspace-specific native probe factory or a Cachet built-in
factory plus backend-native delegate. Use the Python helper below for
release-safe provider-backed targets that also require runtime preflight
sidecars.
The Cachet wheel supplies the runner contract, release-evidence schema, and
built-in `vllm_kv_injection`/`sglang_kv_injection` adapter modules.

Run bundle commands from this folder because `databricks.yml` is the bundle root.

```bash
databricks bundle validate \
  --var runner_python_file=dbfs:/benchmarks/run_engine_probe.py \
  --var handoff_json=/Volumes/catalog/schema/volume/probes/vllm-handoff.json \
  --var probe_factory=document_kv_cache.native_probe_factories:vllm_native_probe_factory \
  --var probe_output_json=/Volumes/catalog/schema/volume/probes/vllm-engine-probe.json \
  --var actions_output_json=/Volumes/catalog/schema/volume/probes/vllm-connector-actions.json \
  --var payload_uri=/Volumes/catalog/schema/volume/probes/vllm-payload.kv \
  --var expected_backend=vllm \
  --var vllm_native_probe_delegate_factory=vllm_kv_injection.probe:build_native_connector_probe \
  --var native_probe_metadata=vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector \
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --var single_user_name=user@example.com
```

The template intentionally omits debug-only `--allow-non-native-probe` and
caller-supplied `--engine-version`. Release evidence expects each native factory
to report the real serving-engine version through `EngineKVProbeFactoryResult`.
The bundle also writes the required
`document_kv.engine_kv_connector_actions.v1` reserve/copy/bind/release
descriptor sidecar through `actions_output_json`.
When `probe_factory` uses Cachet's built-in reserved vLLM or SGLang factory
path, set the matching `vllm_native_probe_delegate_factory` or
`sglang_native_probe_delegate_factory` variable. The bundle injects those values
as `DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY` and
`DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY`; empty defaults are treated as unset
by the built-in factories.
When the delegate is the adapter-package wrapper
`vllm_kv_injection.probe:build_native_connector_probe` or
`sglang_kv_injection.probe:build_native_connector_probe`, set
`native_probe_metadata` to the matching connector factory metadata:
`vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector`
for the built-in provider-backed vLLM path, or
`sglang_kv_injection.connector_factory=sglang_kv_injection.probe:build_document_kv_hicache_probe_connector`
for the built-in provider-backed SGLang HiCache path. Custom runtime patches may
still replace the SGLang value with their own connector factory.
For release-safe provider-backed vLLM or SGLang jobs, prefer the Python helper
below so the target JSON can include the required runtime preflight sidecars; the
runner validates each preflight before starting the native probe.

For non-bundle release operations, the package CLI can emit one `runs/submit`
payload with both required native backend probes:

```bash
python -m document_kv_cache.databricks_engine_probe_job \
  --backend-config-json engine-probe-targets.json \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json databricks-engine-probes-submit.json
```

The target JSON must contain exactly one `vllm` probe and one `sglang` probe in
`--release-safe` mode, and release-safe targets must include
`actions_output_json` for each backend.
Provider-backed vLLM targets must also include
`vllm_runtime_preflight_output_json` and
`vllm_runtime_preflight_layer_names_json`; `document_kv_cache.benchmark_plan`
accepts matching `--engine-probe-vllm-runtime-preflight-*` flags when generating
the target file.
Release-safe SGLang targets must include `sglang_runtime_preflight_output_json`
and `sglang_runtime_preflight_launch_config_json`; the benchmark-plan CLI
accepts matching `--engine-probe-sglang-runtime-preflight-*` flags.
`document_kv_cache.benchmark_plan --engine-probe-targets-output-json ...` can
generate that target JSON from the same planned probe artifacts consumed by
release evidence.
Target records can carry `native_probe_delegate_factory` when the planned
factory is one of Cachet's built-in reserved factory paths.
Release-safe target JSON generated or read by the Python helper rejects
placeholder connector factory metadata such as `module:factory` before writing
or submitting a Databricks payload.
