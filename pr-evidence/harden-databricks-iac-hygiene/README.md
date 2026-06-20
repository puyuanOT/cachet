# Harden Databricks IaC Hygiene

This PR extends repository hygiene so Databricks bundle state, Terraform provider
state, and `terraform.tfstate` files are ignored locally and rejected if they
ever become visible to the hygiene checker.

## Verification

- `poetry run pytest tests/test_repository_hygiene.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`
- `git diff --check`

## Review

GPT-5.5 found that the intended diff was still uncommitted during the first
review pass, which would have produced an empty PR. The branch now commits the
reviewed changes before PR creation; the reviewer reported no code or test
correctness findings in the working-tree diff.
