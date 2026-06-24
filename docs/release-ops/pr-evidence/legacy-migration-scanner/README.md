# Legacy Migration Scanner PR Evidence

This folder records PR #437 evidence for the legacy compatibility migration
scanner. The change adds a scan-config workflow that produces
`document_kv.legacy_compatibility_migration.v1` evidence from checked
downstream runner files while preserving compatibility for existing v1
sidecars.

The GPT-5.5 review first found a same-version schema-compatibility issue. The
PR was updated so `checked_paths` and `legacy_reference_hits` are optional v1
scan provenance, then the reviewer rechecked the branch and reported no
remaining findings. GitHub rejected formal approval because repository owners
cannot approve their own pull requests.
