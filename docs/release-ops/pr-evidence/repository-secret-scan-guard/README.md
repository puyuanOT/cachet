# Repository Secret Scan Guard

This PR-evidence sidecar covers the governance-test slice that adds a lightweight
credential-pattern scan over committed repository text files.

The guard complements ignore rules and contributor documentation by failing the
test suite when source, docs, workflow, or evidence text contains token-shaped
Databricks, OpenAI, GitHub, LangSmith, or PEM private-key material.
