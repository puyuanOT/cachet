# KV Storage Layout Contract Evidence

This sidecar covers the PR slice that adds explicit KV storage-layout semantics to the document KV engine handoff contract.

## Review

GPT-5.5 review found three contract gaps in the first pass:

- `storage_layout` was serialized in handoff records but not bound to persisted cache chunk metadata.
- release evidence required `storage_layout` to exist but did not enforce the V1 Qwen3 shared-storage meaning.
- README wording did not clearly separate storage-family metadata from exact byte-order semantics.

A second review found two remaining consistency issues:

- `qwen3-v1` could still be consistently mislabeled as separate storage if chunks and handoff layout agreed.
- the synthetic storage benchmark used `qwen3-v1` for arbitrary bytes.

The final GPT-5.5 pass was clean after fixing those findings.

## Verification

- `poetry run pytest tests/test_kvpack.py tests/test_storage.py tests/test_storage_benchmark.py tests/test_workflow.py tests/test_model_profiles.py tests/test_engine.py tests/test_engine_adapters.py tests/test_engine_probe.py tests/test_release_evidence.py tests/test_public_package.py -q`
- `poetry run pytest -q`
- `poetry check`
- `python -m py_compile src/restaurant_kv_serving/engine_protocol.py src/restaurant_kv_serving/engine_adapters.py src/restaurant_kv_serving/model_profiles.py src/restaurant_kv_serving/release_evidence.py src/restaurant_kv_serving/models.py src/restaurant_kv_serving/kvpack.py src/restaurant_kv_serving/workflow.py src/restaurant_kv_serving/engine.py src/restaurant_kv_serving/storage_benchmark.py src/document_kv_cache/engine_protocol.py src/document_kv_cache/model_profiles.py`
- `poetry build`
