# PR Evidence Sidecar Key Guards

This PR-evidence sidecar covers the release-bundle schema-hardening slice for
pull-request traceability evidence.

The slice rejects unsupported top-level keys in `document_kv.pr_evidence.v1`
records before copying PR review and Refactor-skill proof into a V1 release
bundle.
