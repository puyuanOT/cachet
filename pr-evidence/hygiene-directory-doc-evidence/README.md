# Directory Documentation Hygiene Evidence

This evidence supports the PR that adds directory documentation coverage to
repository hygiene sidecars and release bundle validation.

The branch records every non-generated tracked or exposed untracked directory
checked by repository hygiene, reports directories missing a `README.md` or
package docstring, and makes release bundles reject sidecars with any missing
directory documentation.

Verification:

- `poetry run pytest tests/test_repository_hygiene.py tests/test_release_bundle.py tests/test_project_governance.py -q`
- `git diff --check && poetry run pytest -q`
- `poetry check && poetry install --dry-run && poetry build`
- GPT-5.5 review requested generated/tooling directory exclusions; the branch
  was patched and re-reviewed cleanly.
