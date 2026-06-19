# Directory Documentation Gate

This PR-evidence sidecar covers the documentation-governance slice that makes
packaged template roots self-documenting and keeps the directory documentation
gate explicit.

The slice adds READMEs for the packaged template resource roots, updates the
template-resource manifest tests to include those files in wheel extraction, and
documents why the governance helper scans tracked plus untracked directories
during local PR work.

## Review

GPT-5.5 review found two low-severity documentation/governance notes:

- the directory documentation gate scans untracked files, which is intentional
  for local PR slices but was not documented in the helper;
- the packaged template README could imply the template package does not need
  to remain importable by `importlib.resources`.

Both findings were addressed: the helper now documents the untracked-directory
behavior, and the template README now states that templates are importable
resources rather than executable runtime modules.

## Verification

- `poetry run pytest tests/test_project_governance.py::test_repository_directories_have_readme_or_package_docstring tests/test_project_governance.py::test_packaged_template_root_readmes_explain_subfolders tests/test_template_resources.py -q`
- `poetry run pytest tests/test_project_governance.py tests/test_template_resources.py -q`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry build`
