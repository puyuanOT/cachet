# Strict PR Evidence Record Schema

This PR-evidence sidecar covers the parser-level schema hardening for
`document_kv.pr_evidence.v1` records.

The slice makes `document-kv-pr-evidence` reject unsupported top-level keys and
shares the same internal key set with release-bundle validation so traceability
evidence cannot be accepted by one gate and rejected by the next.
