# Validate workflow adapter IDs

## Summary

- Added a workflow helper for non-empty, unique string tuples.
- Enforced unique adapter IDs for `TrainingArtifacts`, adapter artifact IDs, and workflow engine handoff adapter IDs.
- Left non-adapter string tuple validation unchanged.

## Why

The engine handle and connector descriptors require adapter IDs to be unique. Workflow-level training artifacts and explicit engine handoff adapter IDs should fail with the same invariant before generation or serving plans are produced.

## Refactor Evidence

- Applied the Refactor skill.
- Extracted unique adapter-ID validation instead of duplicating checks across training and handoff paths.

## Verification

- `poetry run pytest tests/test_workflow.py::test_training_artifacts_validate_adapter_artifact_identity tests/test_workflow.py::test_workflow_engine_handoff_rejects_bare_string_adapter_ids -q`
  - `2 passed`
- `poetry run pytest tests/test_workflow.py -q`
  - `74 passed`
- `poetry run pytest -q`
  - `1230 passed`
- `poetry check`
  - `All set!`
- `poetry run python -m compileall -q src tests`
- `git diff --check`

## Review

- GPT-5.5 subagent approved with no findings.
- Reviewer spot checks:
  - focused workflow adapter-ID tests: `2 passed`
  - full `tests/test_workflow.py`: `74 passed`
