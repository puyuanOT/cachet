# Databricks Bundle Templates

This folder contains Databricks Asset Bundle templates for managed benchmark
execution. The templates are intentionally workspace-agnostic: they do not embed
workspace URLs, tokens, catalogs, or upload paths.

- `databricks.yml` defines the V1 document KV-cache benchmark job on a
  single-node AWS `g6`/L4 GPU cluster.
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
same AWS g6/L4 run to append the storage-reader benchmark after the V1 inference
benchmark.
When generating `v1-plan.json` for the provider-backed vLLM native probe path,
include:
`--engine-probe-metadata vllm=vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector`.

```bash
document-kv-databricks-job \
  --plan-json-uri dbfs:/benchmarks/v1-plan.json \
  --runner-python-file dbfs:/benchmarks/run_plan.py \
  --runner-script-output run_plan.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --execution-result-json-uri dbfs:/benchmarks/result.json \
  --vllm-native-probe-delegate-factory vllm_kv_injection.probe:build_native_connector_probe \
  --sglang-native-probe-delegate-factory my_sglang_adapter.probes:build_probe \
  --output-json runs-submit-reference.json

cd databricks
databricks bundle validate \
  --var plan_json_uri=dbfs:/benchmarks/v1-plan.json \
  --var runner_python_file=dbfs:/benchmarks/run_plan.py \
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --var single_user_name=user@example.com \
  --var execution_result_json_uri=dbfs:/benchmarks/result.json \
  --var transformers_model_id=Qwen/Qwen3-4B-Instruct-2507 \
  --var transformers_device=cuda \
  --var transformers_torch_dtype=bfloat16 \
  --var vllm_native_probe_delegate_factory=vllm_kv_injection.probe:build_native_connector_probe \
  --var sglang_native_probe_delegate_factory=my_sglang_adapter.probes:build_probe
```

The Python helper remains useful for producing a one-off `runs/submit` payload;
the bundle template is the declarative form for teams that manage Databricks jobs
through Asset Bundles. The reference cluster defaults to `SINGLE_USER` access
mode and sets `single_user_name` from `${workspace.current_user.userName}` so
`/Volumes/...` storage-reader evidence uses real Unity Catalog Volume paths. Run
bundle commands from this `databricks/` folder because `databricks.yml` is the
bundle root.
Set the native-probe delegate factory arguments only when the benchmark plan
uses Cachet's built-in reserved vLLM or SGLang probe factories. The helper and
bundle map those values to cluster `spark_env_vars`, leaving the benchmark
runner arguments stable. Empty bundle defaults are treated as unset by the
built-in factories.
Set the `transformers_*` variables only for non-secret generator runtime
configuration, such as model id, device, dtype, or cache-axis order.
The bundle maps them to `CACHET_TRANSFORMERS_*` cluster `spark_env_vars`; leave
them empty when the benchmark plan does not use the Transformers KV generator.

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
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
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
  --wheel-uri /Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
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
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
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
  --wheel-uri /Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json vllm-smoke-runs-submit-reference.json
```

For native engine-probe evidence, generate the probe runner and `runs/submit`
payload after uploading the handoff JSON, payload bytes, package wheel, and
workspace-specific probe factory module:

```bash
document-kv-engine-probe-databricks-job \
  --provider-backed-vllm-native-probe \
  --fixture-output-dir /Volumes/catalog/schema/volume/probes/vllm-fixture \
  --probe-output-json /Volumes/catalog/schema/volume/probes/vllm-engine-probe.json \
  --actions-output-json /Volumes/catalog/schema/volume/probes/vllm-fixture/qwen3-v1-fixture.actions.json \
  --vllm-runtime-preflight-output-json /Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-runtime-preflight.json \
  --vllm-runtime-preflight-layer-names-json /Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-layer-names.json \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --runner-script-output run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json engine-probe-runs-submit-reference.json
```

`--release-safe` rejects debug-only engine-probe options such as
`--allow-non-native-probe` and caller-supplied `--engine-version`; native probe
factories should report the real engine version in their result.
For the built-in vLLM provider-backed path, `--provider-backed-vllm-native-probe`
also sets the Cachet probe factory, vLLM delegate factory, required connector
metadata, expected backend, and pinned `vllm==0.23.0` runtime package.
In release-safe mode it also requires a vLLM runtime preflight output path and
layer-name JSON source, writes the strict preflight record before the native
probe starts, and stops without running the probe if validation fails. It
rejects debug fallback flags and extra wheels so the provider-backed adapter
modules come only from the Cachet wheel.
For the built-in SGLang provider-backed HiCache path,
`--provider-backed-sglang-native-probe` sets the Cachet probe factory, SGLang
delegate factory, required connector metadata, expected backend, and pinned
`sglang==0.5.10.post1` runtime package. In release-safe mode it requires both
SGLang runtime preflight sidecars: `--sglang-runtime-preflight-output-json` and
`--sglang-runtime-preflight-launch-config-json`.
`--actions-output-json` writes the validated reserve/copy/bind/release
descriptor that the native block-manager probe consumed.

Teams that manage these jobs declaratively can instead validate or deploy the
standalone engine-probe bundle:

```bash
cd databricks/engine-probe
databricks bundle validate \
  --var runner_python_file=dbfs:/benchmarks/run_engine_probe.py \
  --var handoff_json=/Volumes/catalog/schema/volume/probes/vllm-handoff.json \
  --var probe_factory=document_kv_cache.native_probe_factories:vllm_native_probe_factory \
  --var probe_output_json=/Volumes/catalog/schema/volume/probes/vllm-engine-probe.json \
  --var actions_output_json=/Volumes/catalog/schema/volume/probes/vllm-connector-actions.json \
  --var payload_uri=/Volumes/catalog/schema/volume/probes/vllm-payload.kv \
  --var expected_backend=vllm \
  --var vllm_native_probe_delegate_factory=vllm_kv_injection.probe:build_native_connector_probe \
  --var native_probe_metadata=vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector \
  --var wheel_uri=/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --var single_user_name=user@example.com
```

The standalone bundle writes the same required connector actions sidecar as the
Python helper through `actions_output_json`. When the probe uses an
adapter-package native wrapper delegate, pass the matching connector factory
through `native_probe_metadata`. Use the Python
`document-kv-engine-probe-databricks-job` helper shown above for two-backend
release-safe probe matrices.

Small runner, wheel, or SGLang launch-config artifacts can be staged to DBFS with the same
environment-provided Databricks credentials before submitting the generated
payload. Use `--require-payload-staged-dbfs-artifacts` for fixture-backed native
probes so generated DBFS output paths do not need fake local upload artifacts.
`stage-and-submit` writes one sidecar with artifact upload provenance and the
Databricks submit response; use a streaming upload tool or a UC Volume for
larger artifacts.

```bash
cachet-databricks-runs \
  --output-json engine-probe-stage-submit-plan.json \
  stage-and-submit \
  --payload-json engine-probe-runs-submit-reference.json \
  --artifact run_engine_probe.py=dbfs:/benchmarks/run_engine_probe.py \
  --require-payload-staged-dbfs-artifacts \
  --dry-run

cachet-databricks-runs \
  --output-json engine-probe-stage-submit-response.json \
  stage-and-submit \
  --payload-json engine-probe-runs-submit-reference.json \
  --artifact run_engine_probe.py=dbfs:/benchmarks/run_engine_probe.py \
  --overwrite \
  --require-payload-staged-dbfs-artifacts
```

If the payload also reads a wheel or SGLang launch config from DBFS, add matching
artifact mappings such as
`--artifact cachet_kv-0.2.0-py3-none-any.whl=dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl`
and
`--artifact sglang-launch-config.json=dbfs:/benchmarks/sglang-launch-config.json`.

Then inspect generated runs from a shell with `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` set:

```bash
cachet-databricks-runs \
  --output-json engine-probe-run-status.json \
  get \
  --run-id 123456789 \
  --summary \
  --submit-payload-json engine-probe-runs-submit-reference.json
```
