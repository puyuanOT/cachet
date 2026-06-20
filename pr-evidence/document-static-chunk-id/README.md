# Document Static Chunk ID

This slice lets document requests name their static context chunk explicitly via
`DocumentKVRequest.static_chunk_id` while preserving the existing `"static"`
default. `CachePlanner` uses that value for `DocumentKVRequest` and keeps the
legacy restaurant request path on the default static chunk.

Verification:

- `poetry run pytest tests/test_planner_materializer.py tests/test_workflow.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

GPT-5.5 review initially caught a positional constructor compatibility bug. The
field was moved after `task_prefix_id`, a regression test was added, and
re-review approved the corrected diff.
