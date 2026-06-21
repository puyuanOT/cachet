# Strict Databricks Purpose Uniqueness

This PR tightens strict V1 release bundle validation for Databricks run-status
sidecars. The bundle already required benchmark, storage, and engine-probe
purpose coverage; it now also rejects duplicate evidence for the same required
purpose.

The new regression test adds a stale duplicate benchmark run-status sidecar and
asserts the strict bundle reports both source files. This prevents old target-run
sidecars from accidentally riding along with the real V1 release evidence.
