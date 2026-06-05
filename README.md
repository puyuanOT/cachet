# Restaurant KV Serving

Serving/cache orchestration for restaurant-review KV reuse.

This repo owns everything outside the inference engine:

- request routing and LoRA/task selection
- Delta/Unity Catalog manifest lookup
- packed KV shard reading
- local NVMe and CPU RAM cache tiers
- CPU-side materialization of selected restaurant static and review chunks
- admission into a vLLM KV-injection connector
- metrics and benchmarks

The vLLM fork/extension should remain thin. It should accept a prepared cache handle from this service and map the already-materialized KV into vLLM's block manager.

## Logical Model

Each restaurant is represented by stable and variable cache chunks:

```text
task_prefix_cache
+ restaurant_static_cache   # description, menu, metadata
+ review_chunk_cache        # selected review IDs only
+ user/task suffix
```

Physical storage uses large packed files, not one file per review:

```text
UC Volume:
  shard_000001.kvpack
  shard_000002.kvpack

Manifest table:
  model_id
  lora_id
  prompt_template_version
  restaurant_id
  chunk_type
  chunk_id
  shard_uri
  byte_offset
  byte_length
  token_count
  dtype
  layout_version
  checksum
```

The current implementation is intentionally storage-format-first and tensor-runtime-agnostic. It materializes ordered byte ranges; the vLLM connector repo owns interpretation as vLLM KV blocks.

## Development

```bash
python -m pip install -e .[test]
pytest -q
```

