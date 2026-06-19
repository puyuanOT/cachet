# Storage Benchmark Bundle

This standalone Databricks Asset Bundle template runs the storage-reader
benchmark on a single AWS g5 node. It produces
`document_kv.storage_benchmark.v1` evidence for memory, local disk, and Unity
Catalog Volume readers.

Run bundle commands from this folder because `databricks.yml` is the bundle
root. Supply `uc_volume_root` as a real `/Volumes/catalog/schema/volume...`
path; release evidence rejects fallback local paths.

```bash
databricks bundle validate \
  --var runner_python_file=dbfs:/benchmarks/run_storage_benchmark.py \
  --var workspace_dir=/local_disk0/document-kv-storage-benchmark \
  --var benchmark_output_json=/Volumes/catalog/schema/volume/storage/storage-benchmark.json \
  --var uc_volume_root=/Volumes/catalog/schema/volume/storage \
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --var single_user_name=user@example.com
```
