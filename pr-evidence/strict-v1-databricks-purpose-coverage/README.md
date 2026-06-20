# Strict V1 Databricks Purpose Coverage

This PR tightens `--require-complete-v1` so a strict release bundle must
include successful Databricks run-status evidence for each target release run
category: the V1 benchmark, storage-reader benchmark, and native engine probe.

The base release-bundle sidecar validation still runs before the strict
purpose check, so invalid or failed Databricks status records keep their
specific release-readiness errors.
