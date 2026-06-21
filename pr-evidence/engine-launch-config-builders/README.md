# Engine Launch Config Builders PR Evidence

PR: https://github.com/puyuanOT/document-kv-cache/pull/295

## What Changed

- Added validated vLLM and SGLang launch-config builders plus a validated JSON writer.
- Added `document-kv-engine-launch-config` and `cachet-engine-launch-config` CLI aliases.
- Exposed the builders through `document_kv_cache`, `cachet`, and legacy `restaurant_kv_serving` compatibility facades.
- Documented strict-release launch-config sidecar generation.

## Review

GPT-5.5 high-reasoning review found one P3 CLI behavior issue: invalid builder inputs raised raw tracebacks. The PR was patched so CLI validation errors route through `argparse` with exit code 2, and a regression test now checks the no-traceback behavior. Delta review reported no new findings.

## Verification

- `PYTHONPATH=src pytest tests/test_engine_launch_config.py tests/test_public_package.py -q`
- `PYTHONPATH=src pytest tests/test_project_governance.py tests/test_pr_evidence.py::test_repository_pr_evidence_sidecars_are_valid -q`
- `PYTHONPATH=src pytest tests/test_release_bundle.py -q -k engine_launch_config`
- `PYTHONPATH=src pytest -q`
- `python -m compileall -q src tests`
- `git diff --check`
- `PYTHONPATH=src python -m document_kv_cache.engine_launch_config build-vllm --extra-config max_model_len=32768`
- `PYTHONPATH=src python -m document_kv_cache.engine_launch_config build-sglang --extra-config deployment='"qa"'`
