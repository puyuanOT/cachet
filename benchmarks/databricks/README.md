# Databricks Benchmark Provenance

The user-facing benchmark tables live in the [benchmark root](../). This
folder no longer mirrors historical Databricks result JSON because those
records used older protocols and confused the public benchmark surface.

For the current protocol, use
[`../appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/`](../appendix/current-q4-q8-vllm-qwen3-4b-g5-a10g/).
Its summary records the Databricks run ids and DBFS output locations.

Do not put raw Jobs API responses, credentials, wheels, driver logs, generated
datasets, or local `databricks-runs/` output here.
