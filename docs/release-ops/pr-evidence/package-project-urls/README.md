# Package Project URLs

This evidence covers the PR that adds public project links to the Cachet
package metadata.

The package now publishes PEP 621 `Repository` and `Issues` URLs for the
`document-kv-cache` GitHub repository. Governance and public-package tests pin
those URLs so future packaging edits keep the published metadata discoverable.

Verification:

- `poetry run pytest tests/test_project_governance.py tests/test_public_package.py -q -k "project_metadata_exposes_repository or project_metadata_uses_cachet_brand or public_package_name"`
- `poetry run pytest`
- `poetry check`
- `poetry build`
- `git diff --check`
- repository secret scan
- GPT-5.5 focused review with no merge-blocking findings
