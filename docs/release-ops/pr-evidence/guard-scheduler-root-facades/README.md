# PR Evidence: guard-scheduler-root-facades

## What changed

- Added a public-package guardrail that keeps `scheduler` out of the `document_kv_cache` and `cachet` root facades.
- Preserved compatibility proof that `document_kv_cache.scheduler` still imports and reuses the admission queue symbols.
- Isolated fresh facade checks in a subprocess so prior explicit compatibility imports cannot make the test order-sensitive.

## Verification

- `poetry run pytest tests/test_public_package.py::test_scheduler_compatibility_module_stays_out_of_root_facades tests/test_scheduler.py tests/test_project_governance.py::test_document_package_readme_lists_public_modules_and_console_scripts tests/test_project_governance.py::test_readme_remaining_work_keeps_serving_boundary_explicit -q`
- `PYTHONPATH=src poetry run pytest -q tests/test_scheduler.py::test_scheduler_namespace_remains_compatible tests/test_public_package.py::test_scheduler_compatibility_module_stays_out_of_root_facades`
- `poetry run pytest tests/test_public_package.py tests/test_project_governance.py tests/test_scheduler.py -q`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry run python -m compileall -q src tests`

## Review

GPT-5.5 reviewer Bernoulli the 3rd found an order-sensitive first version. The
test was refactored to isolate fresh root-facade assertions in a subprocess, and
Bernoulli approved the updated branch.
