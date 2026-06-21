# Validate Release Prompt Text Modes

## Scope

- Require `baseline_prefill` V1 measurements to declare `metadata.prompt_text_mode`
  as `logical`.
- Require `document_kv_cache` V1 measurements to declare
  `metadata.prompt_text_mode` as `runtime`.
- Preserve existing prompt-token-source flexibility while making the benchmark
  prompt representation explicit per arm.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_wrong_prompt_text_mode_for_arm tests/test_release_evidence.py::test_evaluate_release_evidence_requires_prompt_token_context_metadata tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
