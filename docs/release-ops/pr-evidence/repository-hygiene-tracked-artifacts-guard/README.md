# Repository Hygiene Tracked Artifacts Guard

This PR adds a regression guard for the repository hygiene requirement: ignored
local artifacts must not become tracked files. The guard checks `.gitignore`
coverage for build outputs, caches, local environment files, and credential
patterns, then rejects tracked files matching those generated or local-only
patterns while preserving `.env.example` as the documented template exception.

Verification:

- `poetry run pytest tests/test_repository_hygiene.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry build`

GPT-5.5 review found false negatives for env/credential files, wheels/sdists,
and `.pyd` files. Those gaps were patched, and final re-review approved the
diff.
