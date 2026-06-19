# Duplicate Dict Key Governance

This PR-evidence sidecar covers the governance slice that prevents silent
overwrites from repeated literal dictionary keys in Python source.

The guard scans `src/` and `tests/` with `ast`, flags duplicate literal keys in
ordinary dict literals, and also flattens literal `**{...}` dict unpacks in
source order. Dynamic keys and dynamic unpacks are ignored because their values
cannot be proven statically.

## Review

GPT-5.5 found that the first scanner missed literal `**{...}` unpacks. The
scanner now includes literal unpacked dict keys in the parent key stream. A
follow-up diagnostic note about duplicate reporting for unpacked literal dicts
was also addressed by skipping nested dict nodes that were already flattened.
The final GPT-5.5 pass was clean.

## Verification

- `poetry run pytest tests/test_project_governance.py::test_duplicate_literal_dict_key_scanner_reports_silent_overwrites tests/test_project_governance.py::test_python_source_files_do_not_repeat_literal_dict_keys -q`
- `poetry run pytest tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry build`
