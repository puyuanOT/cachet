# Cachet Project Metadata

This evidence covers the PR that aligns package metadata with the Cachet brand
while keeping the public Python distribution name `document-kv-cache`.

The project metadata now includes Cachet in the package description and
keywords. Governance and public-package tests pin that relationship so the
brand can be discoverable without renaming the distribution or import package.

Verification:

- `poetry run pytest tests/test_project_governance.py tests/test_public_package.py -q -k "cachet_brand or project_metadata_uses_cachet_brand or public_package_name"`
- `poetry run pytest`
- `poetry check`
- `poetry build`
- `git diff --check`
- repository secret scan
- GPT-5.5 focused review with no merge-blocking findings
