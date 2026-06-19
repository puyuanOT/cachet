# `document_kv_cache`

This is the public import package for Document KV Cache. It provides
document-namespaced modules while delegating implementation to the legacy
`restaurant_kv_serving` package during migration, so new code can import
document-generic APIs while older Databricks jobs and tests continue to run
unchanged.

Public submodules such as `document_kv_cache.benchmark_plan`,
`document_kv_cache.storage`, `document_kv_cache.storage_benchmark`, and
`document_kv_cache.workflow` are real wrapper modules with
`document_kv_cache.*` module identity. New documentation and examples should use
this package name. The legacy package can be removed after downstream jobs
finish migrating.

The package root keeps `__all__` document-first for `from document_kv_cache
import *`; restaurant-specific compatibility aliases such as
`RestaurantKVRequest` remain available as direct attributes during migration,
but are not advertised by root star imports.

## Public Module Map

Most files in this package are thin public wrappers over the implementation
modules in `restaurant_kv_serving`. They intentionally keep module import paths
and CLI targets under the document-generic namespace. Exported classes and
functions still come from the implementation package during migration, so
tracebacks may include `restaurant_kv_serving` frames.

- `admission.py` exposes pending GPU-memory admission controls.
- `benchmark_plan.py` emits reproducible V1 dataset, benchmark, storage,
  native engine-probe, and release-evidence command plans.
- `benchmark_plan_executor.py` runs command-plan JSON files for local or managed
  job runners and records source-plan SHA-256 provenance.
- `benchmark_runner.py` executes baseline and cache arms against caller-provided
  or OpenAI-compatible engines.
- `benchmarks.py` defines V1 dataset specs, prompt partitioning, measurements,
  summaries, and baseline comparisons.
- `cache.py` exposes CPU and local-disk cache tiers.
- `databricks_engine_probe_job.py` emits a Databricks run payload for native
  vLLM/SGLang engine-probe evidence.
- `databricks_job.py` emits the full V1 benchmark Databricks run payload.
- `databricks_runs.py` submits, checks, and summarizes generated Databricks run
  payloads using environment-provided credentials.
- `databricks_storage_benchmark_job.py` emits standalone storage-reader
  benchmark run payloads.
- `databricks_vllm_smoke_job.py` emits standalone Qwen3/vLLM smoke run payloads.
- `dataset_prep.py` normalizes Biography, HotpotQA, MusiQue, and NIAH data.
- `engine.py` builds engine-ready payload handles from materialized KV.
- `engine_adapters.py` defines the external vLLM/SGLang adapter handoff and
  native-probe descriptor contracts.
- `engine_probe.py` executes a serialized handoff through a backend-owned native
  probe factory.
- `engine_protocol.py` defines validated KV layout, segment, and handle data
  structures.
- `kvpack.py` writes and reads packed KV shard byte ranges.
- `live_server.py` runs a one-request live smoke check against an existing
  OpenAI-compatible serving endpoint.
- `manifest.py` defines manifest lookup and in-memory manifest storage.
- `materializer.py` loads planned chunks into merged or segmented payloads.
- `model_profiles.py` defines model attention geometry, portable JSON profile
  artifacts, and Qwen3 layout helpers.
- `models.py` defines cache keys, chunk references, request models, and
  materialization plans.
- `native_probe_factories.py` exposes reserved vLLM/SGLang native probe factory
  paths plus fail-closed backend environment diagnostics.
- `openai_compatible.py` provides the stdlib-only streaming benchmark engine for
  vLLM/SGLang OpenAI-compatible APIs.
- `planner.py` orders manifest chunks for runtime requests.
- `pr_evidence.py` emits and validates machine-checkable PR traceability,
  Refactor-skill, and GPT-5.5 review evidence for the project workflow.
- `release_bundle.py` copies validated release evidence, optional benchmark
  plan execution records, Databricks run-status records, package wheels, and
  PR-evidence sidecars into a checksummed durable bundle.
- `release_evidence.py` validates V1 benchmark, storage, and native engine-probe
  artifacts, including the pinned serving-engine package/version metadata.
- `service.py` combines planning, materialization, admission, and engine handoff.
- `serving_env.py` records pinned one-engine-per-environment install profiles
  for vLLM and SGLang helpers.
- `storage.py` provides Memory, Disk, Unity Catalog Volume, and routed range
  readers.
- `storage_benchmark.py` measures storage-reader latency and throughput.
- `template_resources.py` lists and extracts packaged templates.
- `vllm_smoke.py` runs the self-contained Qwen3/vLLM Databricks smoke.
- `workflow.py` coordinates optional training, cache generation, manifest
  registration, materialization, and serving preparation.

`_reexport.py` is the private helper used by these wrappers and is not part of
the public API.

## Compatibility-Only Modules

- `scheduler.py` remains packaged as a compatibility shim for older
  admission-helper imports. It is intentionally not advertised through
  `document_kv_cache._PUBLIC_SUBMODULES`.

## Console Scripts

The public package owns these document-branded CLI entry points:

- `document-kv-benchmark-plan`
- `document-kv-run-benchmark-plan`
- `document-kv-databricks-job`
- `document-kv-databricks-runs`
- `document-kv-storage-benchmark`
- `document-kv-storage-benchmark-databricks-job`
- `document-kv-templates`
- `document-kv-release-evidence`
- `document-kv-release-bundle`
- `document-kv-pr-evidence`
- `document-kv-engine-probe`
- `document-kv-engine-probe-databricks-job`
- `document-kv-vllm-smoke`
- `document-kv-vllm-smoke-databricks-job`

Legacy `restaurant-kv-*` aliases remain packaged only for downstream migration.

## Optional Extras

The package keeps Databricks and test dependencies optional but exactly pinned.
Use `databricks` for managed Databricks helpers and `test` for local
verification. vLLM and SGLang are intentionally not Poetry extras because
current releases pin incompatible Torch/Transformers stacks; install the target
engine in a separate serving environment. `serving_env.py` exposes the pinned
helper profiles used by smoke/probe jobs.

Both `document_kv_cache` and the migration-only `restaurant_kv_serving` package
ship `py.typed` markers so downstream type checkers can consume the inline type
annotations from installed wheels.

## Template Subfolders

`templates/` contains package-data templates retrievable from installed wheels.
The `templates/databricks/` subtree mirrors the repository Databricks Asset
Bundle templates for the full V1 benchmark, standalone storage benchmark,
native engine probe, and vLLM smoke jobs.
