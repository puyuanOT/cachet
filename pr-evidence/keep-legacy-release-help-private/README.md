# PR Evidence: keep-legacy-release-help-private

## What changed

- Imported the strict V1 release-bundle help text into the legacy release-bundle facade under a private name.
- Kept the legacy CLI parser wired to the canonical public help text.
- Updated the regression test to prove the legacy module does not expose `STRICT_V1_RELEASE_HELP`.

## Verification

- `poetry run pytest tests/test_release_bundle.py tests/test_public_package.py -q`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry run python -m compileall -q src tests`
- `poetry run pytest tests/test_release_bundle.py::test_release_bundle_strict_release_help_stays_shared_between_public_and_legacy_clis tests/test_release_bundle.py::test_release_bundle_cli_help_documents_strict_release_requirements -q`

## Review

GPT-5.5 reviewer Goodall the 3rd approved with no findings.
