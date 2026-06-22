# Engine Probe Bundle Template

This packaged Databricks Asset Bundle template mirrors
`databricks/engine-probe/databricks.yml` from the repository. It runs one native
vLLM or SGLang engine-probe job on the target AWS g6/L4 Databricks runtime. Use
the Python helper for release-safe provider-backed targets that also require
runtime preflight sidecars.

The workspace still supplies the native probe factory module, handoff JSON,
uploaded payload URI, and connector-actions output URI; the Cachet wheel
supplies the runner contract, release-evidence schema, and built-in
`vllm_kv_injection`/`sglang_kv_injection` adapter modules. The bundle writes the
`document_kv.engine_kv_connector_actions.v1` sidecar through
`actions_output_json`. If the probe uses Cachet's built-in reserved vLLM or
SGLang factory path, set the matching delegate variable so the cluster exports
`DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY` or
`DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY`; the empty defaults are treated as
unset by the built-in factories. When the delegate is the adapter-package
wrapper `vllm_kv_injection.probe:build_native_connector_probe` or
`sglang_kv_injection.probe:build_native_connector_probe`, set
`native_probe_metadata` to the matching connector factory metadata:
`vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector`
for the built-in provider-backed vLLM path, or
`sglang_kv_injection.connector_factory=sglang_kv_injection.probe:build_document_kv_hicache_probe_connector`
for the built-in provider-backed SGLang HiCache path. The standalone bundle forwards
`native_probe_metadata` to the runner; custom runtime patches may still replace
that value with their own connector factory. The Python
Databricks helper remains the path for two-backend release-safe probe matrices
and for provider-backed vLLM or SGLang jobs that require strict runtime
preflight fields.
Release-safe target JSON generated or read by that helper rejects
placeholder connector factory metadata such as `module:factory` before writing or
submitting a Databricks payload. Use the benchmark-plan
`--engine-probe-vllm-runtime-preflight-*` and
`--engine-probe-sglang-runtime-preflight-*` flags when generating target JSON
for that path. For SGLang, the launch-config preflight field must point at the
provider-backed HiCache launch config used by the runtime patch; the helper
executes the strict SGLang preflight before the native probe starts.

For fixture-based probes, `document_kv_cache.probe_fixtures` writes the
deterministic payload, handoff JSON, manifest, and
`qwen3-v1-fixture.actions.json` connector-action descriptor before the native
probe runs. Use that actions sidecar to debug reserve/copy/bind/release
translation inside a backend adapter without relying on generated runner state.
