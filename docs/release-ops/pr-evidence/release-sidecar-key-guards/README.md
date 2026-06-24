# Release Sidecar Key Guards

This PR-evidence sidecar covers the release-bundle schema-hardening slice that
treats release-evidence and preflight sidecars as closed JSON records.

The slice rejects unsupported top-level keys before copying release artifacts
into a bundle, keeping the V1 release manifest constrained to auditable sidecar
schemas.
