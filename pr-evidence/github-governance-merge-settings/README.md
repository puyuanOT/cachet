# PR Evidence: github-governance-merge-settings

## What changed

- Added repository merge settings to GitHub governance sidecars.
- Required release-ready governance to allow squash or rebase merging, enable
  GitHub auto-merge, and delete head branches after merge.
- Extended strict release-bundle validation to require a closed, fully populated
  boolean `merge_settings` block.
- Documented that older governance sidecars without merge settings must be
  regenerated before strict release assembly.

## Verification

- `poetry run pytest tests/test_github_governance.py tests/test_release_bundle.py::test_build_release_bundle_rejects_invalid_package_wheel_pr_evidence_or_github_governance tests/test_project_governance.py::test_github_docs_explain_branch_protection_application_and_plan_limit -q`
- `poetry run pytest tests/test_github_governance.py tests/test_release_bundle.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry run python -m compileall -q src tests`

## Review

GPT-5.5 reviewer Carson the 3rd found schema and readiness gaps. The branch was
updated to require all merge-setting keys, validate boolean values, enforce
auto-merge, document/regress old-sidecar rejection, and Carson approved.
