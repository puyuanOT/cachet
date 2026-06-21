# Databricks Submit Summary Evidence

This folder contains PR evidence for hardening compact Databricks run-status
sidecars. The change requires submit-payload summary arrays to match the
task-level values they summarize, which keeps release-bundle evidence tied to
the generated Databricks task payload instead of stale or hand-edited metadata.
