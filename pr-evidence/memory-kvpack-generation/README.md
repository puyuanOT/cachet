# Memory KVPack Generation

This PR makes in-process Memory a first-class generation target for hot
ephemeral document KV shards.

## Verification

- `poetry run pytest tests/test_workflow.py tests/test_kvpack.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

## Review

GPT-5.5 found two injected-service edge cases before approval: generated
memory shards could be written to a reader that the active service materializer
did not use, and explicit memory writers could bypass the active read path. The
workflow now follows the active materializer rule, mirrors generated and
preloaded memory bytes into compatible service readers, rejects incompatible
memory generation up front, and GPT-5.5 approved the updated branch.
