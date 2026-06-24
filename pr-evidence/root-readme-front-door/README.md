# Root README Front Door Evidence

This folder stores PR evidence for the documentation slice that makes the root
README a clearer Cachet front door and clarifies that `.github/README.md` is
metadata documentation, not the project overview.

The change adds a product-first `Start Here` block, present-tense Cachet
language, current benchmark pointers, root navigation links, and governance
coverage for those README promises.

## Review

Franklin the 3rd reviewed the docs-only diff and found three notes:

- the first screen could imply vLLM and SGLang are separate packages bolted on;
- the README still fell into migration-era restaurant wording immediately after
  the new front door;
- the new Start Here links needed explicit target-heading coverage.

The README now says the adapter modules ship with `cachet-kv`, the opening uses
present-tense product language, and governance tests assert the target headings
exist.

## Verification

- `poetry run pytest tests/test_project_governance.py::test_readme_documents_cachet_brand_and_scope tests/test_project_governance.py::test_github_docs_explain_branch_protection_application_and_plan_limit -q`
- `poetry run pytest tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check --lock`
- local README markdown link check
- changed-file secret scan
