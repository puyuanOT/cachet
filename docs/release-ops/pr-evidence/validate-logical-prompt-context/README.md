# Validate Logical Prompt Context

## Scope

- Require matched baseline and cache measurements for the same V1
  dataset/example to report the same `logical_prompt_tokens`.
- Reject repeated measurements for the same dataset/example/arm when their
  logical prompt-token counts disagree.
- Preserve the existing per-arm runtime-token checks that distinguish full
  prefill from cache-reuse serving.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_inconsistent_measurement_logical_prompt_tokens tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_prompt_tokens_that_do_not_match_arm_context tests/test_release_evidence.py::test_evaluate_release_evidence_allows_repeated_raw_measurements -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
