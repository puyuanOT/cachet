# Release Preflight Record-Type Guard

This PR-evidence sidecar covers the release-readiness preflight slice that
separates wrong-role JSON artifacts from missing or unreadable inputs.

The slice adds `invalid_record_type_paths` to
`document_kv.release_evidence_inputs.v1`, so preflight checks reject readable
JSON files that have the wrong `record_type` for the CLI slot they occupy.
