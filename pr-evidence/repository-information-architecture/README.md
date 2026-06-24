# Repository Information Architecture Evidence

This folder stores PR evidence for the documentation slice that makes Cachet's
repository layout easier to navigate without moving or deleting durable release
artifacts.

The change adds a repository map, an evidence policy, root README links, docs
index links, and governance coverage for the benchmark/evidence/PR-evidence
boundaries.

## Review

Banach the 3rd reviewed the docs-only diff and found three notes:

- the newly linked docs needed to be tracked with the PR;
- `docs/repo-map.md` used ambiguous "one package" wording;
- the new policy docs needed governance-test coverage.

The wording now says Cachet is one project and one distribution package, the
new docs are part of the PR, and `tests/test_project_governance.py` pins the
navigation and evidence-policy boundaries.

## Verification

- `poetry run pytest tests/test_project_governance.py::test_repository_map_and_evidence_policy_are_documented -q`
- `poetry run pytest tests/test_project_governance.py -q`
- `poetry check --lock`
- `git diff --check`
- local markdown link check over touched docs
- `poetry run cachet-pr-evidence --validate-directory pr-evidence`
