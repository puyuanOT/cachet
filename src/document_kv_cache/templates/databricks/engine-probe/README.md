# Engine Probe Bundle Template

This packaged Databricks Asset Bundle template mirrors
`databricks/engine-probe/databricks.yml` from the repository. It runs one native
vLLM or SGLang engine-probe job on the target AWS g6/L4 Databricks runtime.

The workspace still supplies the native probe factory module, handoff JSON,
uploaded payload URI, and connector-actions output URI; the package supplies
the runner contract and release-evidence schema. The bundle writes the
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
`sglang_kv_injection.connector_factory=module:factory` for SGLang. The Python
Databricks helper remains the path for two-backend release-safe probe matrices.

For fixture-based probes, `document_kv_cache.probe_fixtures` writes the
deterministic payload, handoff JSON, manifest, and
`qwen3-v1-fixture.actions.json` connector-action descriptor before the native
probe runs. Use that actions sidecar to debug reserve/copy/bind/release
translation inside a backend adapter without relying on generated runner state.
