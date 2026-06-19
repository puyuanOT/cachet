# GitHub Metadata

This folder contains repository-level contribution automation and templates.

- `pull_request_template.md` captures the required PR description, Refactor
  skill evidence, GPT-5.5 review evidence, and test or benchmark evidence for
  each logical development slice.
- `workflows/ci.yml` runs the automated pull-request quality gate and repeats
  the same safety check on pushes to `main`.
