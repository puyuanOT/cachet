# Databricks

Maintainer reference for running benchmark plans, serving smokes, storage
benchmarks, and native engine probes as single-node Databricks jobs, and for
staging, submitting, and checking those runs. Asset Bundle templates, the
storage-benchmark job, and the vLLM smoke job are documented in
[`databricks/README.md`](../../../databricks/README.md). The commands that run
inside these jobs are described in [`benchmarks.md`](benchmarks.md) and
[`serving-engines.md`](serving-engines.md).

## Cluster Rules

- Payload helpers emit a single-node GPU cluster in `SINGLE_USER` access mode
  so jobs can read and write real Unity Catalog Volume paths. The Python
  helpers require `--single-user-name`; the Asset Bundles default it to
  `${workspace.current_user.userName}`.
- `--hardware-target aws-g6-l4` (default, `g6.8xlarge`) or
  `--hardware-target aws-g5-a10g` (`g5.8xlarge`) derives the node type. Pass
  `--node-type-id` only when a cluster policy requires a specific matching
  size.
- Programmatic integrations use `DatabricksSingleNodeGPUClusterConfig`,
  `build_single_node_gpu_cluster`, and `validate_aws_single_node_gpu_type`.
- Use `--spark-env-var` only for non-secret runtime configuration such as the
  Transformers model id, device, dtype, or cache-axis order. The helpers reject
  secret-looking names and Databricks PAT-shaped values.
- Keep generated `runs/submit` payloads, task logs, and status JSONs under the
  ignored `databricks-runs/` directory or inside release bundles; repository
  hygiene fails if that directory is tracked or visible as untracked.
- Datasets and artifacts may use `disk:`, `file:`, `dbfs:/`, `/dbfs/`,
  `/Volumes/`, `uc-volume:/`, or `uc-volume://` paths; the plan runner
  normalizes them on the driver.

## Managed Benchmark Plan

Upload the wheel, the plan JSON from
[`benchmarks.md`](benchmarks.md#benchmark-plan), and a runner script, then
generate the payload. When the plan uses the provider-backed native probe
paths, generate it with
`--engine-probe-metadata vllm=vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector`
and
`--engine-probe-metadata sglang=sglang_kv_injection.connector_factory=sglang_kv_injection.probe:build_document_kv_hicache_probe_connector`
so the plan carries the strict connector-factory metadata before a GPU job is
submitted.

```bash
mkdir -p databricks-runs/managed-plan
python -m document_kv_cache.databricks_job \
  --plan-json-uri dbfs:/benchmarks/v1-plan.json \
  --runner-python-file dbfs:/benchmarks/run_plan.py \
  --runner-script-output databricks-runs/managed-plan/run_plan.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --hardware-target aws-g6-l4 \
  --spark-env-var CACHET_TRANSFORMERS_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 \
  --spark-env-var CACHET_TRANSFORMERS_DEVICE=cuda \
  --spark-env-var CACHET_TRANSFORMERS_TORCH_DTYPE=bfloat16 \
  --vllm-native-probe-delegate-factory vllm_kv_injection.probe:build_native_connector_probe \
  --sglang-native-probe-delegate-factory my_sglang_adapter.probes:build_probe \
  --output-json databricks-runs/managed-plan/databricks-run-submit.json
```

- Set the native-probe delegate flags only when the plan uses Cachet's
  built-in reserved vLLM or SGLang probe factories. They become cluster
  `spark_env_vars` (`DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY` and
  `DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY`) so plan execution arguments stay
  stable.
- With `--execution-result-json-uri`, the result record includes a
  `plan_source` block (original and driver path, byte size, SHA-256,
  suite/model/hardware fields, and command count) that ties execution back to
  the uploaded plan.

## Serving Smoke Jobs

`cachet-vllm-smoke-databricks-job` accepts the same prepared-dataset, handoff
generation, and sizing flags as `cachet-vllm-smoke` and installs the uploaded
wheel into both the driver and the isolated vLLM environment; see
[`databricks/README.md`](../../../databricks/README.md) for the payload command.

For SGLang, the preferred handoff-backed readiness payload uses
`--generate-live-handoff` plus non-secret Transformers runtime variables:

```bash
mkdir -p databricks-runs/sglang-smoke
python -m document_kv_cache.databricks_sglang_smoke_job \
  --benchmark-id v1_sglang_live_001 \
  --output-dir /Volumes/catalog/schema/volume/cachet-sglang-live \
  --runner-python-file dbfs:/benchmarks/run_sglang_smoke.py \
  --runner-script-output databricks-runs/sglang-smoke/run_sglang_smoke.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --hardware-target aws-g6-l4 \
  --generate-live-handoff \
  --live-handoff-output-dir /Volumes/catalog/schema/volume/cachet-sglang-live/live-handoff \
  --spark-env-var CACHET_TRANSFORMERS_DEVICE=cuda \
  --spark-env-var CACHET_TRANSFORMERS_TORCH_DTYPE=bfloat16 \
  --output-json databricks-runs/sglang-smoke/databricks-sglang-smoke-submit.json
```

Use `--baseline-only` for provider and server bring-up, or pass
`--handoff-json`/`--handoff-record-json` plus `--sglang-hicache-page-keys-json`
to reuse a validated handoff. Prepared SGLang dataset runs default to the
Qwen3 V1 page size (`16`) unless `--sglang-hicache-page-size` is set; add
`--benchmark-handoff-generator-factory` and `--benchmark-handoff-output-dir` to
generate handoffs and enriched JSONL inside the isolated SGLang runtime, where
they pass the same page-size and page-key coverage gate before server start.

Both packaged runners have passed against the vLLM 0.30.0 serving profile on
both V1 hardware targets. The vLLM runner passed on L4 and A10G on 2026-11-18
(Databricks runs `597966250980954` and `473099143422393`). The SGLang
live-handoff runner passed on A10G on 2026-11-18 (`621605817274575`) and on L4
on 2026-11-19 (`274263423127999`). In both SGLang runs the quality canary and
both arms returned the expected answer, and after a cache flush the cache
request reused 204 of its 244 prompt tokens from HiCache. `aws-g6-l4` accepts
any `g6.` node type; the L4 run used `g6.4xlarge` because `g6.8xlarge` and
`g6.16xlarge` had no AWS capacity that day.

## Engine-Probe Jobs

For release runs, generate one two-backend target file with the benchmark
planner (`--engine-probe-targets-output-json --engine-probe-targets-release-safe`)
and submit one release-safe payload so vLLM and SGLang exercise the same
descriptor contract on the same cluster policy:

```bash
mkdir -p databricks-runs/engine-probe-matrix
python -m document_kv_cache.databricks_engine_probe_job \
  --backend-config-json /data/engine-probe-targets.json \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --runner-script-output databricks-runs/engine-probe-matrix/run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json databricks-runs/engine-probe-matrix/databricks-engine-probes-submit.json
```

- Target files are a closed schema; unsupported top-level or per-probe keys
  are rejected before a payload is written.
- In matrix mode, `--release-safe` requires exactly one native vLLM task and
  one native SGLang task and rejects debug fallbacks (`engine_version`
  overrides, `allow_non_native_probe`). Every release-safe target must carry
  `actions_output_json` and `native_probe_factories_output_json`; the runner
  writes the factory diagnostics inside that backend's isolated runtime after
  any runtime preflight and before the probe, and stops if diagnostics fail.
- Provider-backed vLLM targets must carry `vllm_runtime_preflight_output_json`
  and `vllm_runtime_preflight_layer_names_json`; release-safe SGLang targets
  must carry `sglang_runtime_preflight_output_json` and
  `sglang_runtime_preflight_launch_config_json`. Generate them with the
  planner's `--engine-probe-vllm-runtime-preflight-*` and
  `--engine-probe-sglang-runtime-preflight-*` flags. The runner executes the
  strict preflight first and returns its nonzero exit code without starting
  the probe. `dbfs:/...` preflight paths are normalized to `/dbfs/...` on the
  driver; inline JSON values pass through unchanged.
- `native_probe_delegate_factory` (planner flag
  `--engine-probe-native-delegate-factory BACKEND=MODULE:CALLABLE`) is mapped
  to the matching `DOCUMENT_KV_*_NATIVE_PROBE_FACTORY` cluster variable rather
  than a runner argument.
- Release-safe provider-backed targets get their exact runtime packages from
  `document_kv_cache.serving_env`. A target `pip_packages` entry for the same
  package must match that pin exactly (for example `sglang==0.5.10.post1`);
  use `pip_packages` only for additional backend-specific packages. The vLLM
  runtime is the reviewed E5M2-patched `0.30.0+cu129` wheel plus the packaged
  hash lock, so set `DOCUMENT_KV_VLLM_PATCHED_WHEEL_URI` and
  `DOCUMENT_KV_VLLM_PATCHED_WHEEL_SHA256` when generating vLLM probe payloads.
- Fixture-backed targets (`fixture_output_dir`, `fixture_payload_mode`) write
  the deterministic Qwen3 fixture and its connector-action sidecar on the
  target node before the probe runs. Release-safe fixtures must use the native
  adapter payload mode (`merged`).
- Add `--serial-tasks` to run one backend cluster at a time when GPU capacity
  or cost requires it.
- Packages and wheels are installed into a driver venv
  (`/local_disk0/cachet/engine-probe-venv` by default) and the probe re-execs
  inside it, so Databricks ML image packages do not leak into the serving
  runtime.

Single-backend QA probes use `--provider-backed-vllm-native-probe` or
`--provider-backed-sglang-native-probe`, which set the Cachet probe factory,
the adapter delegate, the connector metadata, the expected backend, and the
pinned runtime. In release-safe mode they require
`--native-probe-factories-output-json` plus the backend's runtime preflight
flags, and they reject debug fallback flags and extra wheels. Without
`--task-key`, release-safe single-backend jobs use
`document_kv_engine_probe_vllm` or `document_kv_engine_probe_sglang`, which
strict bundles expect.

## Submitting And Checking Runs

Dry-run a payload locally before any credentials or network requests are
involved:

```bash
cachet-databricks-runs \
  --output-json databricks-runs/managed-plan/databricks-run-submit-summary.json \
  payload-summary \
  --payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --expected-hardware-target aws-g6-l4 \
  --expected-node-type-id g6.8xlarge

cachet-databricks-runs \
  --output-json databricks-runs/managed-plan/databricks-stage-submit-plan.json \
  stage-and-submit \
  --payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --artifact v1-plan.json=dbfs:/benchmarks/v1-plan.json \
  --artifact databricks-runs/managed-plan/run_plan.py=dbfs:/benchmarks/run_plan.py \
  --require-payload-dbfs-artifacts \
  --dry-run
```

Use `--require-payload-staged-dbfs-artifacts` instead for payloads that write
generated DBFS outputs during the run, such as fixture-backed probes; it still
checks DBFS runner, wheel, plan, and SGLang launch-config inputs.

Credentials come from `DATABRICKS_HOST` and `DATABRICKS_TOKEN`, or from
`--profile PROFILE_NAME` (with `--config-file PATH` outside the default
`~/.databrickscfg`). Never put tokens in arguments or JSON artifacts.

- Profiles with `host` and `token` are parsed directly. Profiles with
  `auth_type` and no static token resolve through the Databricks SDK, which
  needs the `databricks` extra.
- `--profile-auth-mode sdk` forces SDK auth for OAuth or Databricks CLI
  profiles that also hold a transient static token; `--profile-auth-mode static`
  fails unless a static token is present.
- `auth-check` makes a non-mutating current-user request and records only the
  endpoint, HTTP status, response key names, and a SHA-256 of the host.
  `stage-and-submit --preflight-auth-check` runs it after local artifact
  validation and before any upload or submission.

```bash
cachet-databricks-runs \
  --profile QA \
  --output-json databricks-runs/managed-plan/databricks-stage-submit-response.json \
  stage-and-submit \
  --payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --artifact v1-plan.json=dbfs:/benchmarks/v1-plan.json \
  --artifact databricks-runs/managed-plan/run_plan.py=dbfs:/benchmarks/run_plan.py \
  --overwrite \
  --require-payload-dbfs-artifacts \
  --preflight-auth-check

cachet-databricks-runs \
  --output-json databricks-runs/managed-plan/databricks-run-status.json \
  get \
  --run-id 123456789 \
  --summary \
  --submit-payload-json databricks-runs/managed-plan/databricks-run-submit.json \
  --expected-hardware-target aws-g6-l4 \
  --expected-node-type-id g6.8xlarge
```

`get --summary` writes a compact `document_kv.databricks_run_status.v1` block
(lifecycle and result states, active task key, cluster ids, timing, and
success/terminal flags) plus a sanitized submit-payload hash and cluster
summary when `--submit-payload-json` is given. `--include-response` adds the
raw `runs/get` payload for debugging only; release bundles reject it.
Preflight status sidecars with `validate_databricks_run_status_sidecar` or
`databricks_run_status_sidecar_issues`, passing
`expected_hardware_target="aws-g6-l4"` and `expected_node_type_id="g6.8xlarge"`
for strict V1 checks so stale or non-default-size sidecars cannot be reused.
