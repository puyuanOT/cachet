# Release Bundle PR Evidence Review Gate

This PR-evidence sidecar covers the release-bundle traceability slice that
locks the PR-evidence review gate into both the Python API and public CLI paths.

The slice documents that release bundles only accept valid PR evidence sidecars
with Refactor-skill evidence, completed GPT-5.5 review, and resolved GPT-5.5
findings, then adds regression coverage for a sidecar that incorrectly claims
`ok: true` while leaving review findings unresolved.
