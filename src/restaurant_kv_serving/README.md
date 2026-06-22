# `restaurant_kv_serving`

This package contains migration shims for callers that have not yet moved to
`document_kv_cache`. New code should import `document_kv_cache`. This package
remains available for existing Databricks jobs and tests during migration.

Except for `scheduler.py`, which is an older admission-helper shim, modules in
this package forward to document-owned implementations while preserving legacy
module paths, root exports, and selected monkeypatch hooks.

- `models.py` defines cache keys, chunk references, document requests, and materialization plans.
- `manifest.py` defines manifest lookup interfaces and an in-memory implementation for tests.
- `kvpack.py` writes and reads packed KV shard byte ranges.
- `engine.py` defines engine-ready KV handles, tensor layouts, and the vLLM/SGLang connector protocol.
- `engine_adapters.py` is a compatibility facade over `document_kv_cache.engine_adapters`, which owns the external vLLM and SGLang adapter contracts and validation probes without importing or replacing those serving engines.
- `engine_probe.py` runs a serialized engine handoff through a backend-provided native probe factory and writes release-gate probe evidence JSON.
- `probe_fixtures.py` is a compatibility wrapper over `document_kv_cache.probe_fixtures`, which writes deterministic Qwen3 V1 handoff fixtures for native vLLM/SGLang connector development.
- `model_profiles.py` records model attention geometry and derives validated KV layouts for MHA, GQA, and MQA-style caches.
- `storage.py` defines Memory, Disk, Unity Catalog Volume, and routed range readers.
- `cache.py` implements CPU and local-disk byte cache tiers, including hit/miss stats, checksum-safe local paths, and optional local disk byte budgets.
- `materializer.py` loads selected chunks and assembles either merged or segmented payloads.
- `planner.py` translates a document request into ordered manifest segments.
- `admission.py` admits prepared requests under a pending GPU-memory budget before handoff to vLLM, SGLang, or another serving engine.
- `scheduler.py` is a compatibility shim for old admission-helper imports.
- `service.py` combines planning, materialization, and admission for serving integrations.
- `workflow.py` coordinates source documents, optional training, cache generation, method labels such as vanilla prefill or KV Packet, manifest registration, materialization, and engine-ready serving preparation.
- `serving_env.py` records pinned one-engine-per-environment install profiles for vLLM and SGLang helpers.
- `benchmarks.py` is a compatibility wrapper over `document_kv_cache.benchmarks`, which owns the V1 dataset specs, deterministic prompt/context builders, and quality/latency schema for comparing baseline prefill against document KV-cache reuse.
- `dataset_prep.py` is a compatibility wrapper over `document_kv_cache.dataset_prep`, which owns Biography, HotpotQA, MusiQue, and NIAH normalization into the JSONL schema consumed by `benchmark_runner.py`, including synthetic NIAH generation.
- `benchmark_plan.py` emits reproducible dataset-preparation and benchmark-runner command plans for target AWS g6/L4/Qwen3 V1 jobs, while the Databricks helpers enforce supported V1 GPU node families. The default/release target remains AWS g6/L4, with explicit AWS g5/A10G compatibility runs available under `aws-g5-a10g`.
- `benchmark_handoffs.py` is a compatibility wrapper over `document_kv_cache.benchmark_handoffs`, which generates per-row Cachet handoff bundles and joins those artifacts onto prepared V1 benchmark JSONL rows before Databricks benchmark execution.
- `benchmark_plan_executor.py` executes a benchmark plan JSON command sequence, primarily for managed job runners.
- `databricks_job.py` is a compatibility wrapper for the document-owned AWS g6/L4 Databricks V1 benchmark job payload helper.
- `databricks_storage_benchmark_job.py` is a compatibility wrapper for the document-owned AWS g6/L4 Databricks storage-reader job payload helper.
- `databricks_engine_probe_job.py` is a compatibility wrapper for the document-owned AWS g6/L4 Databricks native vLLM/SGLang engine-probe job payload helper.
- `databricks_runs.py` is a compatibility wrapper over `document_kv_cache.databricks_runs`, which stages small DBFS artifacts, can dry-run or stage-and-submit generated Databricks payloads, and checks run state using only `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables.
- `benchmark_runner.py` is a compatibility wrapper over `document_kv_cache.benchmark_runner`, which owns canonical V1 JSONL loading, caller-provided or OpenAI-compatible vLLM/SGLang benchmark execution, and JSON measurement, summary, and comparison records.
- `release_evidence.py` is a compatibility wrapper over `document_kv_cache.release_evidence`, which validates collected V1 benchmark, storage benchmark, and native vLLM/SGLang probe JSON artifacts before a release is called complete, and records the input artifact sources in the final release-evidence JSON.
- `openai_compatible.py` is a compatibility wrapper over `document_kv_cache.openai_compatible`, which owns the thin streaming completion engine for vLLM/SGLang OpenAI-compatible API servers.
- `live_server.py` is a compatibility wrapper over `document_kv_cache.live_server`, which owns the one-request live smoke check against an existing OpenAI-compatible vLLM/SGLang endpoint and prints a JSON latency/quality record.
- `storage_benchmark.py` writes a synthetic packed shard and reports Memory, Disk, and Unity Catalog reader latency/throughput plus selected-reader and strict release machine-checkable evidence under configurable parallel read load.
- `vllm_smoke.py` is a compatibility wrapper over `document_kv_cache.vllm_smoke`, which owns the isolated Databricks-local vLLM environment, starts Qwen3 4B Instruct, and runs a tiny V1 Biography/HotpotQA/MusiQue/NIAH smoke through the OpenAI-compatible benchmark runner.

`openai_compatible.py` posts full logical prompts by default, which is the correct behavior for ordinary OpenAI-compatible vLLM/SGLang servers and platform-managed prefix caching. Its `prompt_text_mode="runtime"` option is only for a KV-aware adapter or proxy that binds cached prefixes out of band and expects the runtime suffix in the `prompt` field.

`benchmark_plan.py` emits the dataset-preparation and benchmark commands that should run on the target AWS g6/L4/Qwen3 environment:

```bash
python -m document_kv_cache.benchmark_plan \
  --raw-dataset biography=/raw/biography.jsonl \
  --raw-dataset hotpotqa=/raw/hotpotqa.jsonl \
  --raw-dataset musique=/raw/musique.jsonl \
  --raw-dataset niah=/raw/niah.jsonl \
  --prepared-dir /data/v1-prepared \
  --base-url http://localhost:8000 \
  --storage-benchmark-workspace-dir /local_disk0/document-kv-storage-benchmark \
  --storage-benchmark-uc-volume-root /Volumes/catalog/schema/volume/document-kv-storage-benchmark \
  --engine-probe-handoff-json vllm=/data/vllm-handoff.json \
  --engine-probe-output-json vllm=/data/vllm-engine-probe.json \
  --engine-probe-actions-output-json vllm=/data/vllm-connector-actions.json \
  --engine-probe-handoff-json sglang=/data/sglang-handoff.json \
  --engine-probe-output-json sglang=/data/sglang-engine-probe.json \
  --engine-probe-actions-output-json sglang=/data/sglang-connector-actions.json \
  --engine-probe-use-builtin-factories \
  --release-evidence-output-json /data/release-evidence.json \
  --plan-output-json /data/v1-plan.json \
  --plan-output-sh /data/run-v1-benchmark.sh
```

When storage benchmark flags are present, the plan appends
`document_kv_cache.storage_benchmark` after the inference benchmark so Memory
and Disk reader latency/throughput are measured on the same AWS g6/L4 node. The
Unity Catalog reader is included only when a real `--storage-benchmark-uc-volume-root`
is provided. Backend-keyed `--engine-probe-*` options append native
`document_kv_cache.engine_probe` commands for vLLM and SGLang handoffs; release
evidence automatically consumes those planned probe and connector-action
outputs. Planned probes consumed by release evidence must include
`--engine-probe-actions-output-json` for each backend and cannot use debug-only
`--engine-probe-engine-version` or
`--allow-non-native-engine-probe`; use explicit `--release-engine-probe-json`
and `--release-engine-actions-json` records when a plan also contains debug
probe commands. Existing native probe and connector-action JSONs can still be
supplied directly with repeatable `--release-engine-probe-json` and
`--release-engine-actions-json`. When
`--engine-probe-use-builtin-factories` is present, missing factories are filled
with package-owned vLLM/SGLang factory paths. The vLLM path can use Cachet's
provider-backed delegate from the wheel with strict connector metadata and
runtime preflight, while SGLang still requires a backend-native delegate.
When release-evidence flags are present, the plan appends `document_kv_cache.release_evidence`
last so the V1 benchmark, storage benchmark, and native vLLM/SGLang engine probe
JSON artifacts are validated together. Release evidence also checks each probe's
runtime engine version and serving-engine package/version metadata against the
pinned profile in `serving_env.py`. The same validator supports `--preflight-only` and
`--preflight-output-json` to report missing or unreadable evidence files before
the strict release gate runs. `dataset_prep.py` and `benchmark_runner.py` also expose direct CLIs
for ad hoc local use:

```bash
python -m document_kv_cache.dataset_prep \
  --dataset hotpotqa \
  --input-jsonl /data/raw_hotpotqa.jsonl \
  --output-jsonl /data/v1_hotpotqa.jsonl
```

For synthetic NIAH smoke data, omit `--input-jsonl` and provide `--haystack-text` or `--haystack-file` plus `--needle-answer`.

```bash
python -m document_kv_cache.benchmark_runner \
  --dataset biography=/data/biography.jsonl \
  --dataset hotpotqa=/data/hotpotqa.jsonl \
  --dataset musique=/data/musique.jsonl \
  --dataset niah=/data/niah.jsonl \
  --base-url http://localhost:8000 \
  --output-json v1-results.json
```

The CLI preserves the same baseline-vs-cache prompt split as the programmatic API and is intended for target AWS g6/L4/Qwen3 4B Instruct V1 runs.
OpenAI-style bases ending in `/v1` are normalized before appending the default `/v1/completions` endpoint. Custom `--endpoint` or `--cache-endpoint` values are appended to the exact base URL for servers that expose a different completions route. Use `--cache-runtime-prompt` only together with an explicit `--cache-base-url`.

To smoke-test the real vLLM server path inside a Databricks GPU task, use:

```bash
python -m document_kv_cache.vllm_smoke \
  --benchmark-id v1_vllm_smoke_001 \
  --output-dir /Volumes/catalog/schema/volume/document-kv-v1-smoke
```

The smoke writes the server import probe, server log, benchmark report, and
metadata into `--output-dir`. It is a reproducibility check for the pinned
vLLM/Qwen3 environment, not a replacement for the full release benchmark plan.
The import probe also instantiates Cachet's `DocumentKVConnector` from the
server `KVTransferConfig` and fails unless the configured provider factory
resolves to native document-KV wiring.
When all four prepared dataset paths are supplied, the smoke requires every row
to carry Cachet `kv_transfer_params`, writes `prepared-handoff-coverage.json`,
and runs the cache arm with the logical prompt so the native provider path is
exercised against vLLM's connector scheduler.

To emit a Databricks `runs/submit` payload for that same smoke task, use:

```bash
python -m document_kv_cache.databricks_vllm_smoke_job \
  --benchmark-id v1_vllm_smoke_001 \
  --output-dir /Volumes/catalog/schema/volume/document-kv-v1-smoke \
  --runner-python-file dbfs:/benchmarks/run_vllm_smoke.py \
  --runner-script-output run_vllm_smoke.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json databricks-vllm-smoke-submit.json
```

To produce native vLLM/SGLang connector evidence from a handoff JSON, run:

```bash
python -m document_kv_cache.engine_probe \
  --handoff-json req-123-handoff.json \
  --probe-factory my_engine_adapter.probes:build_probe \
  --expected-backend vllm \
  --actions-output-json vllm-connector-actions.json \
  --output-json vllm-engine-probe.json
```

The factory owns the real vLLM or SGLang block-manager calls;
`document_kv_cache.engine_probe` owns payload loading, descriptor validation,
and the machine-checkable probe record, including the serving-engine profile
metadata consumed by release evidence.
The optional `--actions-output-json` sidecar writes the validated
`document_kv.engine_kv_connector_actions.v1` descriptor used by that native
probe.

To run the same native probe through a Databricks-managed AWS g6/L4 task, generate
the small runner script and `runs/submit` payload:

```bash
python -m document_kv_cache.databricks_engine_probe_job \
  --provider-backed-vllm-native-probe \
  --fixture-output-dir /Volumes/catalog/schema/volume/probes/vllm-fixture \
  --probe-output-json /Volumes/catalog/schema/volume/probes/vllm-engine-probe.json \
  --actions-output-json /Volumes/catalog/schema/volume/probes/vllm-fixture/qwen3-v1-fixture.actions.json \
  --vllm-runtime-preflight-output-json /Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-runtime-preflight.json \
  --vllm-runtime-preflight-layer-names-json /Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-layer-names.json \
  --runner-python-file dbfs:/benchmarks/run_engine_probe.py \
  --runner-script-output run_engine_probe.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --release-safe \
  --output-json databricks-engine-probe-submit.json
```

For Cachet's built-in provider-backed vLLM path,
`--provider-backed-vllm-native-probe` fills in the Cachet probe factory, vLLM
delegate factory, strict connector metadata, expected backend, and pinned
`vllm==0.23.0` runtime package.
In release-safe mode it also requires the vLLM runtime preflight output path and
layer-name JSON source so the Databricks runner can validate the installed vLLM
contract and layer mapping before starting the native probe. It rejects debug
fallback flags and extra wheels so the provider-backed adapter modules come only
from the Cachet wheel.

Use `--release-safe` for release-evidence jobs so debug-only non-native probes
and caller-supplied engine versions are rejected before the Databricks payload is
written.

For Databricks-managed execution, first upload the package wheel, the generated benchmark plan JSON, and the tiny runner script. Then emit a single-node AWS g6/L4 `runs/submit` payload:

When generating the referenced plan JSON for the provider-backed vLLM native
probe path, include
`--engine-probe-metadata vllm=vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector`.

```bash
python -m document_kv_cache.databricks_job \
  --plan-json-uri dbfs:/benchmarks/v1-plan.json \
  --runner-python-file dbfs:/benchmarks/run_plan.py \
  --runner-script-output run_plan.py \
  --wheel-uri dbfs:/benchmarks/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --vllm-native-probe-delegate-factory vllm_kv_injection.probe:build_native_connector_probe \
  --sglang-native-probe-delegate-factory my_sglang_adapter.probes:build_probe \
  --output-json databricks-run-submit.json
```

Set the native-probe delegate factory flags only when the benchmark plan uses
Cachet's built-in reserved vLLM or SGLang native probe factories. The helper
maps those paths to cluster environment variables, so benchmark runner
parameters stay stable across serving environments.

The payload helper intentionally keeps authentication separate. To submit or
inspect a run from a machine with env-provided Databricks credentials, use:

```bash
export DATABRICKS_HOST=https://dbc-...cloud.databricks.com
export DATABRICKS_TOKEN=...
python -m document_kv_cache.databricks_runs \
  --output-json databricks-submit-response.json \
  submit \
  --payload-json databricks-run-submit.json

python -m document_kv_cache.databricks_runs \
  --output-json databricks-run-status.json \
  get \
  --run-id 123456789 \
  --summary
```

External CI, Databricks Asset Bundles, or workspace-specific scripts can still
POST generated payloads to `/api/2.1/jobs/runs/submit` directly.

To measure storage-reader load on the same hardware, use:

```bash
python -m document_kv_cache.storage_benchmark \
  --workspace-dir /local_disk0/document-kv-storage-benchmark \
  --uc-volume-root /Volumes/catalog/schema/volume/document-kv-storage-benchmark \
  --reader memory \
  --reader disk \
  --reader unity_catalog \
  --output-json storage-benchmark.json
```

To run that storage benchmark as a Databricks-managed AWS g6/L4 task, use:

```bash
python -m document_kv_cache.databricks_storage_benchmark_job \
  --workspace-dir /local_disk0/document-kv-storage-benchmark \
  --benchmark-output-json /Volumes/catalog/schema/volume/storage/storage-benchmark.json \
  --uc-volume-root /Volumes/catalog/schema/volume/storage \
  --runner-python-file dbfs:/benchmarks/run_storage_benchmark.py \
  --runner-script-output run_storage_benchmark.py \
  --wheel-uri /Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl \
  --single-user-name user@example.com \
  --output-json databricks-storage-benchmark-submit.json
```

The restaurant-specific names are compatibility aliases. New code should prefer document-generic APIs and avoid adding new restaurant-specific surface area.
`KVCacheKey` stores `document_id` as its canonical identifier, while
`restaurant_id` remains a constructor/property alias for migration. Likewise,
`MaterializationPlan.selected_document_ids` is canonical and
`selected_restaurants` is retained only as a compatibility property.
