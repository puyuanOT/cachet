# Strict V1 Release Bundle Completeness

This PR adds a strict release-publishing mode for `document_kv_cache.release_bundle`.
The default bundle behavior remains permissive for compatibility, while
`--require-complete-v1` now fails unless the V1 bundle includes the required
release evidence, preflight, plan execution, Databricks run-status, tested
wheel, PR evidence, GitHub governance, repository hygiene, and native-probe
factory diagnostics artifacts.

GPT-5.5 review initially found that generated benchmark plans cannot bundle
their own final plan-execution sidecar because the executor writes that record
after the plan completes. The fix keeps strict completeness on the direct
release-bundle API/CLIs and documents the release flow as two phases: execute
the benchmark plan first, then build the strict bundle with the completed
`plan-execution.json`.
