# Engine Probe Bundle Template

This packaged Databricks Asset Bundle template mirrors
`databricks/engine-probe/databricks.yml` from the repository. It runs one native
vLLM or SGLang engine-probe job on the target AWS g5 Databricks runtime.

The workspace still supplies the native probe factory module, handoff JSON, and
uploaded payload URI; the package supplies the runner contract and
release-evidence schema.
Use the Python Databricks engine-probe job helper when a release job also needs
the optional `document_kv.engine_kv_connector_actions.v1` sidecar.
