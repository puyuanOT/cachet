# Engine Probe Bundle Template

This packaged Databricks Asset Bundle template mirrors
`databricks/engine-probe/databricks.yml` from the repository. It runs one native
vLLM or SGLang engine-probe job on the target AWS g5 Databricks runtime.

The workspace still supplies the native probe factory module, handoff JSON, and
uploaded payload URI; the package supplies the runner contract and
release-evidence schema.
