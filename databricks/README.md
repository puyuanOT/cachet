# Databricks Bundle Templates

This folder contains Databricks Asset Bundle templates for managed benchmark
execution. The templates are intentionally workspace-agnostic: they do not embed
workspace URLs, tokens, catalogs, or upload paths.

- `databricks.yml` defines the V1 document KV-cache benchmark job on a
  single-node AWS `g5` GPU cluster.
- `vllm-smoke/databricks.yml` defines a smaller managed smoke job for the
  self-contained Qwen3/vLLM server check.
- `storage-benchmark/databricks.yml` defines a standalone Memory/Disk/Unity
  Catalog Volume storage-reader benchmark job for release evidence.
- `engine-probe/databricks.yml` defines a standalone native vLLM/SGLang
  engine-probe job; each workspace still supplies its own adapter probe factory.

The same bundle templates are included in package artifacts under
`document_kv_cache/templates/databricks/` so release consumers can retrieve them
without cloning the repository.
Use `document-kv-templates list --prefix databricks` to inspect packaged
templates or `document-kv-templates extract --prefix databricks --output-dir
./document-kv-templates` to copy them from an installed wheel.

Prepare the benchmark plan, wheel, and tiny runner script with the package CLIs,
upload those artifacts using your workspace tooling, then validate or deploy the
bundle with explicit variables:

Include `--storage-benchmark-workspace-dir` and, when available,
`--storage-benchmark-uc-volume-root` while generating the plan if you want the
same AWS g5 run to append the storage-reader benchmark after the V1 inference
benchmark.

```bash
document-kv-databricks-job \
  --plan-json-uri dbfs:/benchmarks/v1-plan.json \
  --runner-python-file dbfs:/benchmarks/run_plan.py \
  --runner-script-output run_plan.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --execution-result-json-uri dbfs:/benchmarks/result.json \
  --output-json runs-submit-reference.json

cd databricks
databricks bundle validate \
  --var plan_json_uri=dbfs:/benchmarks/v1-plan.json \
  --var runner_python_file=dbfs:/benchmarks/run_plan.py \
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --var single_user_name=user@example.com \
  --var execution_result_json_uri=dbfs:/benchmarks/result.json
```

The Python helper remains useful for producing a one-off `runs/submit` payload;
the bundle template is the declarative form for teams that manage Databricks jobs
through Asset Bundles. The reference cluster defaults to `SINGLE_USER` access
mode and sets `single_user_name` from `${workspace.current_user.userName}` so
`/Volumes/...` storage-reader evidence uses real Unity Catalog Volume paths. Run
bundle commands from this `databricks/` folder because `databricks.yml` is the
bundle root.

For the smallest managed runtime check, use the standalone smoke bundle or
generate the equivalent one-off `runs/submit` payload. The smoke bundle only
requires the wheel, the smoke runner script, a benchmark id, and an output
directory:

```bash
cd databricks/vllm-smoke
databricks bundle validate \
  --var runner_python_file=dbfs:/benchmarks/run_vllm_smoke.py \
  --var benchmark_id=v1_vllm_smoke_001 \
  --var output_dir=/Volumes/catalog/schema/volume/document-kv-v1-smoke \
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --var single_user_name=user@example.com
```

The standalone smoke bundle is the preferred first GPU check because it does not
require full V1 raw datasets, prepared benchmark plans, or storage-result paths.
Run bundle commands from `databricks/vllm-smoke/` because that folder is the
smoke bundle root.

For standalone storage-reader evidence, generate the storage runner and
`runs/submit` payload after uploading the package wheel. Use a real
`/Volumes/...` root for UC evidence:

```bash
document-kv-storage-benchmark-databricks-job \
  --workspace-dir /local_disk0/document-kv-storage-benchmark \
  --benchmark-output-json /Volumes/catalog/schema/volume/storage/storage-benchmark.json \
  --uc-volume-root /Volumes/catalog/schema/volume/storage \
  --runner-python-file dbfs:/benchmarks/run_storage_benchmark.py \
  --runner-script-output run_storage_benchmark.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json storage-benchmark-runs-submit-reference.json
```

Teams that manage this job declaratively can instead validate or deploy the
standalone storage benchmark bundle:

```bash
cd databricks/storage-benchmark
databricks bundle validate \
  --var runner_python_file=dbfs:/benchmarks/run_storage_benchmark.py \
  --var workspace_dir=/local_disk0/document-kv-storage-benchmark \
  --var benchmark_output_json=/Volumes/catalog/schema/volume/storage/storage-benchmark.json \
  --var uc_volume_root=/Volumes/catalog/schema/volume/storage \
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --var single_user_name=user@example.com
```

For an ad hoc managed smoke without Asset Bundles, generate the separate smoke
runner and `runs/submit` payload:

```bash
document-kv-vllm-smoke-databricks-job \
  --benchmark-id v1_vllm_smoke_001 \
  --output-dir /Volumes/catalog/schema/volume/document-kv-v1-smoke \
  --runner-python-file dbfs:/benchmarks/run_vllm_smoke.py \
  --runner-script-output run_vllm_smoke.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json vllm-smoke-runs-submit-reference.json
```

For native engine-probe evidence, generate the probe runner and `runs/submit`
payload after uploading the handoff JSON, payload bytes, package wheel, and
workspace-specific probe factory module:

```bash
document-kv-engine-probe-databricks-job \
  --handoff-json /Volumes/catalog/schema/volume/probes/vllm-handoff.json \
  --probe-factory my_engine_adapter.probes:build_probe \
  --probe-output-json /Volumes/catalog/schema/volume/probes/vllm-engine-probe.json \
  --actions-output-json /Volumes/catalog/schema/volume/probes/vllm-connector-actions.json \
  --payload-uri /Volumes/catalog/schema/volume/probes/vllm-payload.kv \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --runner-script-output run_engine_probe.py \
  --expected-backend vllm \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json engine-probe-runs-submit-reference.json
```

`--release-safe` rejects debug-only engine-probe options such as
`--allow-non-native-probe` and caller-supplied `--engine-version`; native probe
factories should report the real engine version in their result.
`--actions-output-json` writes the validated reserve/copy/bind/release
descriptor that the native block-manager probe consumed.

Teams that manage these jobs declaratively can instead validate or deploy the
standalone engine-probe bundle:

```bash
cd databricks/engine-probe
databricks bundle validate \
  --var runner_python_file=dbfs:/benchmarks/run_engine_probe.py \
  --var handoff_json=/Volumes/catalog/schema/volume/probes/vllm-handoff.json \
  --var probe_factory=my_engine_adapter.probes:build_probe \
  --var probe_output_json=/Volumes/catalog/schema/volume/probes/vllm-engine-probe.json \
  --var actions_output_json=/Volumes/catalog/schema/volume/probes/vllm-connector-actions.json \
  --var payload_uri=/Volumes/catalog/schema/volume/probes/vllm-payload.kv \
  --var expected_backend=vllm \
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --var single_user_name=user@example.com
```

The standalone bundle writes the same required connector actions sidecar as the
Python helper through `actions_output_json`. Use the Python
`document-kv-engine-probe-databricks-job` helper shown above for two-backend
release-safe probe matrices.

Then submit or inspect generated payloads from a shell with `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` set:

```bash
document-kv-databricks-runs \
  --output-json vllm-smoke-submit-response.json \
  submit \
  --payload-json vllm-smoke-runs-submit-reference.json

document-kv-databricks-runs \
  --output-json vllm-smoke-run-status.json \
  get \
  --run-id 123456789 \
  --summary \
  --submit-payload-json vllm-smoke-runs-submit-reference.json
```
