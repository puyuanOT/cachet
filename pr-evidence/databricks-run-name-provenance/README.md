# Databricks Run Name Provenance

This PR tightens Databricks run-status sidecar validation. When both the fetched
run summary and attached submit-payload provenance include a `run_name`, they
must match.

That prevents a successful Databricks run-status sidecar from being paired with
a stale submit payload for a different run while still preserving compatibility
for records that omit optional run names.
