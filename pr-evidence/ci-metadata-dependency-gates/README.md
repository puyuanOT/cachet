# CI Metadata Dependency Gates

This PR-evidence sidecar covers the CI hardening slice that adds package
metadata validation and dependency-resolution dry runs before the repository's
test and build steps.

The slice keeps the existing Poetry-based CI structure and adds gates for
`poetry check`, the base dependency set, and the optional `databricks` plus
`test` dependency set.
