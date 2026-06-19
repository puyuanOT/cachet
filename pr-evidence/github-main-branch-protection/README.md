# GitHub Main Branch Protection

This PR-evidence sidecar covers the governance slice that documents and tests
the GitHub `main` branch protection payload.

The slice adds an operator-ready REST payload for `main`, documents how to apply
it, records the private-repository branch-protection plan limitation observed
during setup, and adds governance tests that keep the required status-check name
tied to the actual CI workflow job.

## Review

GPT-5.5 found two low-severity governance issues:

- the branch-protection test asserted the required CI check string without
  proving it matched the actual workflow job name;
- the README did not mention that required linear history needs repository merge
  settings that keep squash or rebase merging enabled.

Both findings were addressed. The test now extracts the workflow job name and
asserts it is uniquely `Test and build`, and the README documents the
linear-history merge-setting prerequisite. The final GPT-5.5 review found no
remaining findings.

## Verification

- `poetry run pytest tests/test_project_governance.py::test_github_main_branch_protection_payload_requires_pr_review_and_ci tests/test_project_governance.py::test_github_docs_explain_branch_protection_application_and_plan_limit -q`
- `python -m json.tool .github/main-branch-protection.json >/tmp/main-branch-protection.pretty.json && python -m py_compile tests/test_project_governance.py`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry build`
- `PYTHONPATH=src python -m document_kv_cache.pr_evidence --validate-directory pr-evidence --output-json /tmp/pr-evidence-main-protection-validation.json`
