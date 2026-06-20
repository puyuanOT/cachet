# Engine Probe Bundle Template

This packaged Databricks Asset Bundle template mirrors
`databricks/engine-probe/databricks.yml` from the repository. It runs one native
vLLM or SGLang engine-probe job on the target AWS g5 Databricks runtime.

The workspace still supplies the native probe factory module, handoff JSON,
uploaded payload URI, and connector-actions output URI; the package supplies
the runner contract and release-evidence schema. The bundle writes the
`document_kv.engine_kv_connector_actions.v1` sidecar through
`actions_output_json`, while the Python Databricks helper remains the path for
two-backend release-safe probe matrices.
