# Plan Execution Command Details

Strict V1 release bundles reject benchmark plan execution sidecars unless each command
looks like an executed command: a non-empty name, a non-empty argv array, `skipped:
false`, `returncode: 0`, and `error: null`.

Verification:

- `poetry run pytest tests/test_release_bundle.py::test_build_release_bundle_plan_execution_stays_out_of_release_sidecar_matching tests/test_release_bundle.py::test_build_release_bundle_rejects_plan_execution_sidecars_with_extra_keys tests/test_release_bundle.py::test_build_release_bundle_rejects_plan_execution_command_count_mismatch tests/test_release_bundle.py::test_build_release_bundle_rejects_non_executed_plan_execution_commands -q`
- `poetry run pytest tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry run python -m compileall -q src tests`
