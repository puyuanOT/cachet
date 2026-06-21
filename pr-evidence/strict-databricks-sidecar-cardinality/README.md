# Strict Databricks Sidecar Cardinality

This PR tightens strict V1 release bundle validation for target Databricks run
evidence. A strict bundle must now include exactly one run-status sidecar for
each required purpose: benchmark, storage benchmark, and engine probe.

The validator also rejects a single multi-task sidecar that tries to satisfy
multiple required purposes at once. The regression test builds that combined
sidecar shape and verifies the bundle reports both the artifact count mismatch
and the multi-purpose sidecar.
