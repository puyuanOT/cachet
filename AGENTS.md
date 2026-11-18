# AGENTS.md

Cachet (`cachet-kv`, imported as `cachet`) stores, loads, and hands off reusable
document KV-cache payloads to vLLM and SGLang, which keep scheduling, decode,
and native GPU KV blocks.

## Layout And Tooling

- `src/document_kv_cache/` is the implementation; `src/cachet/` is the branded
  re-export facade; `src/vllm_kv_injection/` and `src/sglang_kv_injection/` are
  the vendored engine adapters.
- `benchmarks/` holds sealed published results; `databricks/` holds Asset
  Bundle templates.
- Python 3.11 or 3.12 and Poetry 2.4.1. Direct dependencies are exact pins in
  `pyproject.toml`; keep `poetry.lock` in sync.
- vLLM and SGLang are not project dependencies; serving runtimes run in
  isolated environments pinned by `document_kv_cache.serving_env`.

## Commands

```bash
poetry install -E test
poetry run python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
poetry run pytest -q
poetry run pytest tests/test_project_governance.py -q
poetry check --lock
python benchmarks/verify_sealed_records.py
```

- CI runs `poetry run ruff check` and `poetry run mypy` only on the files listed
  in `.github/workflows/ci.yml`; run both on any listed file you change.
- Torch tests skip without the CPU wheel. Without Poetry, use
  `python -m pip install -e '.[test]'` and `pytest tests -q`.

## Rules

- Keep Cachet at the engine handoff boundary; do not add a scheduler, decode
  loop, or native block manager.
- Use `cachet` imports and `cachet-*` commands in new docs; every console script
  also keeps its `document-kv-*` alias.
- Never commit credentials, raw service responses, wheels, logs, or
  `databricks-runs/` output; see `docs/evidence-policy.md`. Do not hand-edit
  sealed records under `benchmarks/`.
- `tests/test_project_governance.py` pins doc phrases; when moving doc text,
  point the assertion at the new file.

More: [`CONTRIBUTING.md`](CONTRIBUTING.md), [maintainer topics](docs/release-ops/maintainer-reference.md).
