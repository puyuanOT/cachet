# Databricks Benchmark Provenance

No Databricks benchmark result is currently published in this directory.
Sanitized results of the Databricks campaign are in
[`../appendix/`](../appendix/).

The vLLM 0.27.1 campaign design, run on vLLM 0.30.0, reserves each workload
before submission and binds
the exact payload digest, returned run identities, terminal control-plane
status, timestamps, duration, hardware, and result closure. Only sanitized,
sealed records whose benchmark-root regions are validated byte-for-byte may be
added to [`../appendix/`](../appendix/).

Never commit credentials, raw Jobs API responses, wheels, driver logs,
generated datasets, prompt payloads, or local `databricks-runs/` output.
