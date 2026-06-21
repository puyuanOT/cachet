# Databricks Purpose Sidecar Evidence

This folder contains PR evidence for hardening release-oriented Databricks
run-status sidecars. The change requires each submit-payload task summary to
carry a non-empty `purpose` tag so strict V1 release bundles can audit managed
run coverage from compact status evidence.
