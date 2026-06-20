# Engine GPU Estimate Validation

This directory contains the machine-readable validation summary for the PR that
hardens engine-ready GPU byte estimate validation.

Verification:

- `poetry run pytest -q tests/test_engine.py tests/test_workflow.py`
- `poetry run pytest -q tests/test_public_package.py::test_public_document_submodules_have_curated_star_import_surfaces`
- `python -m py_compile src/document_kv_cache/engine.py tests/test_engine.py`
- `git diff --check`
- `poetry run pytest -q`
- `poetry check`
- repository hygiene
- GPT-5.5 focused review
