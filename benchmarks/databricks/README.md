# Databricks Benchmark Provenance

No Databricks benchmark result is currently published in this directory.

The vLLM 0.27.1 campaign must reserve each workload before submission and bind
the exact payload digest, returned run identities, terminal control-plane
status, timestamps, duration, hardware, and result closure. Only sanitized
records that pass the campaign publication gate may be added to
[`../appendix/`](../appendix/).

Never commit credentials, raw Jobs API responses, wheels, driver logs,
generated datasets, prompt payloads, or local `databricks-runs/` output.
