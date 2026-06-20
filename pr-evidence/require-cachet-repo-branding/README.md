# Require Cachet Repository Branding

## What Changed

- GitHub governance readiness now requires a non-empty repository description that mentions Cachet.
- GitHub governance readiness now requires repository topics to include `cachet` and `kv-cache`.
- Release-bundle validation independently rejects GitHub governance sidecars that lack the Cachet-branded description or required topics.
- README governance instructions now document these branding requirements.

## Why

The project goal requires distinct premium package branding and a clear repository description for open-source release readiness. The release sidecars should fail closed when repository metadata does not advertise Cachet and document KV-cache discovery topics.

## Scope

- `README.md`
- `src/document_kv_cache/github_governance.py`
- `src/document_kv_cache/release_bundle.py`
- `tests/test_github_governance.py`
- `tests/test_release_bundle.py`
- `pr-evidence/require-cachet-repo-branding/README.md`

## Verification

- `python -m pytest tests/test_github_governance.py tests/test_release_bundle.py -q` -> 53 passed
- `python -m pytest tests/test_github_governance.py tests/test_release_bundle.py tests/test_project_governance.py::test_repository_directories_have_readme_or_package_docstring -q` -> 54 passed
- `python -m pytest -q` -> 872 passed
- `poetry check` -> All set
- `poetry install --dry-run` -> succeeded
- `poetry build` -> succeeded
- `git diff --check` -> clean
- GPT-5.5 review -> README finding fixed, then APPROVED with no findings
