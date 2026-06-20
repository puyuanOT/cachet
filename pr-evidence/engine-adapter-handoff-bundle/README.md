# Engine Adapter Handoff Bundle

## What Changed

- Added `write_engine_adapter_handoff_bundle(...)` to write coordinated engine
  handoff JSON and payload artifacts.
- Exported the helper through `document_kv_cache`, `cachet`, and the legacy
  `restaurant_kv_serving` compatibility namespace.
- Updated the README handoff example to recommend the coordinated writer.
- Added regression coverage for merged, segmented, same-path, equivalent URI,
  and hard-link destination collisions.

## Why

Serving integrations should not need to manually keep payload bytes and JSON
handoff descriptors in sync. A single public helper makes the vLLM/SGLang
handoff path less error-prone while preserving the lower-level APIs for
specialized workflows.

## Verification

- `poetry run pytest tests/test_engine_probe.py -q` -> 34 passed
- `poetry run pytest tests/test_public_package.py tests/test_project_governance.py -q` -> 47 passed
- `poetry run pytest -q` -> 1164 passed
- `poetry check` -> All set
- `git diff --check` -> passed
- `poetry run python -m compileall -q src tests` -> passed
- GPT-5.5 review -> findings resolved, final approval
