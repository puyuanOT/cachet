# Validate Release Prompt Token Meaning

## Scope

- Require `baseline_prefill` V1 measurement `prompt_tokens` to match
  `metadata.logical_prompt_tokens`.
- Require `document_kv_cache` V1 measurement `prompt_tokens` to match
  `metadata.runtime_prompt_tokens`.
- Preserve existing report-row aggregate validation while making each raw
  measurement's token-count meaning explicit.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_prompt_tokens_that_do_not_match_arm_context tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_wrong_prompt_text_mode_for_arm tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
