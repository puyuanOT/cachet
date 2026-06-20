# Packaged Template Extraction Notes

This PR documents how installed-wheel users can list and extract the packaged
Databricks Asset Bundle templates with `document-kv-templates`, then run bundle
commands from the extracted bundle roots.

Verification:

- `poetry run pytest tests/test_template_resources.py -q`
- `git diff --check`
- `poetry run python -m compileall -q src tests`
- `poetry run pytest -q`

GPT-5.5 review outcome: approved with no blocking findings.
