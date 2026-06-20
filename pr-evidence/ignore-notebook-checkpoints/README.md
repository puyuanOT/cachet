# Ignore Notebook Checkpoints

This PR tightens repository hygiene for exploratory Databricks or Jupyter work
by treating `.ipynb_checkpoints` folders as generated artifacts.

## Verification

- `poetry run pytest tests/test_repository_hygiene.py tests/test_project_governance.py -q`
- `git diff --check`
- `poetry run pytest -q`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

## Review

GPT-5.5 requested two fixes: align the governance directory scanner with the
production generated-directory list, and make the repository-hygiene tests pass
checkpoint paths through the evaluator. Both findings were fixed, verification
was rerun, and the reviewer approved the branch.
