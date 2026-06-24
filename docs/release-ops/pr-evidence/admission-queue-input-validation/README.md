# Admission Queue Input Validation

This directory contains the machine-readable validation summary for the PR that
hardens admission queue inputs at the serving handoff boundary.

Verification:

- `poetry run pytest -q tests/test_admission.py tests/test_workflow.py`
- `python -m py_compile src/document_kv_cache/admission.py tests/test_admission.py`
- `git diff --check`
- `poetry run pytest -q`
- `poetry check`
- repository hygiene
- GPT-5.5 focused review
