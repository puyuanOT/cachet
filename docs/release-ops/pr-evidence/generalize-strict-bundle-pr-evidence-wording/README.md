# Generalized Strict Bundle PR Evidence Wording

This folder contains PR evidence for replacing stale PR-number-specific strict
bundle provenance wording with stable release-gate PR evidence wording.

The PR keeps strict bundle provenance auditable without tying the tracked docs
to a previously merged PR number. The release-bundle validator still requires a
valid PR evidence sidecar with pull-request identity, Refactor-skill evidence,
completed GPT-5.5 review evidence, and repository alignment.

Verification:

- `git diff --check`
- `poetry run pytest tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `poetry check && poetry check --lock`
- post-428 ignored strict bundle rebuild with PR #428 evidence sidecar
- GPT-5.5 focused review approved with no findings
