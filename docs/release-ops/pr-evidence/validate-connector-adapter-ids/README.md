# Validate connector adapter IDs

## Summary

- Refactored connector-side adapter ID validation into one helper in `engine_adapters.py`.
- Enforced unique ordered adapter IDs for `EngineKVInjectionPlan`, reservation actions, bind actions, and serialized engine handoff records.
- Kept connector descriptors aligned with `KVCacheHandle`, which already rejects duplicate adapter IDs.

## Why

V1 serving can use LoRA adapters with vLLM and SGLang connector actions. Duplicate adapter IDs in the connector descriptor are ambiguous, can make release evidence look valid while describing impossible adapter state, and diverge from the engine handle contract.

## Refactor Evidence

- Applied the Refactor skill.
- Consolidated duplicated adapter-ID validation without changing unrelated connector action logic.

## Verification

- `poetry run pytest tests/test_engine_adapters.py::test_engine_kv_connector_actions_record_validation_rejects_stale_fields tests/test_engine_adapters.py::test_validate_engine_adapter_request_record_rejects_schema_and_segment_mismatch tests/test_engine_adapters.py::test_engine_kv_connector_actions_reject_duplicate_adapter_ids -q`
  - `3 passed`
- `poetry run pytest tests/test_engine_adapters.py::test_validate_engine_adapter_request_record_rejects_schema_and_segment_mismatch tests/test_engine_adapters.py::test_engine_kv_injection_plan_rejects_invalid_identity_and_sequence_fields tests/test_engine_adapters.py::test_engine_kv_connector_actions_reject_duplicate_adapter_ids tests/test_engine_adapters.py::test_engine_kv_connector_actions_record_round_trips_segmented_handoff -q`
  - `16 passed`
- `poetry run pytest tests/test_engine_adapters.py -q`
  - `125 passed`
- `poetry run pytest -q`
  - `1230 passed`
- `poetry check`
  - `All set!`
- `poetry run python -m compileall -q src tests`
- `git diff --check`

## Review

- GPT-5.5 subagent approved with no blocking findings.
- Reviewer suggested an optional explicit serialized connector-action duplicate-adapter-ID regression; this PR now includes it.
