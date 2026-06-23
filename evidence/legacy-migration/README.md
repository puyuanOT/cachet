# Legacy Migration Evidence

This directory tracks evidence for removing the source-only
`restaurant_kv_serving` compatibility facade from the repository after local
tests and downstream runners no longer require it.

The `current/` folder contains the latest generated
`document_kv.legacy_compatibility_migration.v1` record, its scan config, and a
validation wrapper. Release bundles can attach that record through the optional
`legacy_migration_evidence` artifact role when a package-surface change needs
that proof.
