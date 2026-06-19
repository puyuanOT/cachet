# Secret Hygiene Ignore Rules

This PR-evidence sidecar covers the repository hygiene slice that keeps local
credential files and generated outputs out of version control.

The slice extends `.gitignore`, documents the no-secrets rule for Databricks,
OpenAI, private keys, benchmark records, release evidence, and generated runner
scripts, and adds governance tests that check both required patterns and actual
`git check-ignore` behavior.

