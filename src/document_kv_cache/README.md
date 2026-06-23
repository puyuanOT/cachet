# `document_kv_cache`

This is the compatibility-preserving implementation package for Cachet. It
provides document-namespaced modules that own the implementation while the
branded `cachet` package exposes the user-facing root and `cachet.<module>`
import surface.

Public submodules such as `document_kv_cache.benchmark_plan`,
`document_kv_cache.storage`, `document_kv_cache.storage_benchmark`, and
`document_kv_cache.workflow` are the canonical implementation modules with
`document_kv_cache.*` module identity. New documentation and examples should
prefer the Cachet aliases, while this namespace remains importable for existing
benchmark evidence, Databricks runners, and downstream jobs.

The package root keeps `__all__` document-first for `from document_kv_cache
import *`; restaurant-specific compatibility aliases are no longer exported
from the package surface or retained in the core runtime model layer.

## Public Module Map

Public files in this package define the document-owned classes, functions,
records, and CLI targets. The removed restaurant compatibility package must not
return as a production dependency.

- `admission.py` exposes pending GPU-memory admission controls.
- `adapter_scaffold.py` generates fail-closed native-probe delegate modules for
  backend-specific vLLM/SGLang adapter packages.
- `benchmark_plan.py` emits reproducible V1 dataset, benchmark, storage,
  native engine-probe, release-readiness sidecar, and release-evidence command
  plans.
- `benchmark_handoffs.py` generates per-row Cachet handoff bundles from
  prepared benchmark JSONL, builds fail-closed `(dataset, example_id)` handoff
  manifests, carries optional SGLang HiCache page-key proof, and attaches those
  handoffs to prepared benchmark JSONL rows.
- `benchmark_handoff_bundles.py` is the `python -m` entry point for the
  handoff-bundle generation CLI used by reproducible benchmark plans.
- `benchmark_plan_executor.py` runs command-plan JSON files for local or managed
  job runners and records source-plan SHA-256 provenance.
- `benchmark_runner.py` owns baseline and cache-arm execution against
  caller-provided or OpenAI-compatible engines.
- `benchmarks.py` owns V1 dataset specs, prompt partitioning, cache-prefix
  request helpers, measurements, summaries, and baseline comparisons.
- `cache.py` exposes CPU and local-disk cache tiers.
- `databricks_engine_probe_job.py` emits a Databricks run payload for native
  vLLM/SGLang engine-probe evidence, including backend-specific delegate
  factory environment variables when generated target JSON asks built-in
  reserved probe factories to call downstream native adapters. Release-safe
  targets also run strict vLLM/SGLang runtime preflights before native probe
  execution.
- `databricks_job.py` emits the full V1 benchmark Databricks run payload,
  including optional non-secret generator runtime environment variables and
  native-probe delegate environment variables for benchmark plans that use
  Cachet's built-in vLLM/SGLang probe factories.
- `databricks_runs.py` stages small DBFS artifacts, dry-runs or performs
  stage-and-submit for a generated payload with one provenance sidecar, and
  checks/summarizes Databricks runs using environment variables, static
  `.databrickscfg` token profiles, or optional Databricks SDK-backed OAuth
  profiles. Its CLI can force static-token or SDK-backed profile resolution
  when Databricks CLI/OAuth profiles carry transient static tokens. It also
  provides a non-mutating auth-check command for launch readiness without
  recording tokens or user details, plus a stage-and-submit preflight flag that
  runs that check before DBFS uploads or run submission.
- `databricks_storage_benchmark_job.py` owns standalone storage-reader
  benchmark run payloads.
- `databricks_sglang_smoke_job.py` owns standalone Qwen3/SGLang live smoke run
  payloads that target AWS g5/g6 Databricks clusters.
- `databricks_vllm_smoke_job.py` owns standalone Qwen3/vLLM smoke run payloads.
- `dataset_prep.py` owns Biography, HotpotQA, MusiQue, and NIAH
  normalization into canonical benchmark JSONL.
- `engine.py` builds engine-ready payload handles from materialized KV.
- `engine_adapters.py` defines the external vLLM/SGLang adapter handoff and
  native-probe descriptor contracts.
- `engine_launch_config.py` builds and validates vLLM transfer-config and
  SGLang HiCache launch sidecars against the shared document-KV engine handoff
  contract. Generated vLLM sidecars include Cachet's built-in native provider
  factory so `DocumentKVConnector` is provider-backed by default.
- `engine_probe.py` executes a serialized handoff through a backend-owned native
  probe factory.
- `engine_protocol.py` defines validated KV layout, segment, and handle data
  structures.
- `github_governance.py` emits GitHub repository visibility, branch
  protection, merge-settings, auto-merge, and merged-branch cleanup status
  records for release-readiness evidence.
- `kvpack.py` writes and reads packed KV shard byte ranges.
- `legacy_compatibility.py` validates downstream migration evidence proving the
  legacy `restaurant_kv_serving` compatibility facade was removed safely.
- `live_server.py` owns the one-request live smoke check against an existing
  OpenAI-compatible serving endpoint, including validated Cachet handoff params
  for native vLLM/SGLang cache-arm requests.
- `manifest.py` defines manifest lookup and in-memory manifest storage.
- `materializer.py` loads planned chunks into merged or segmented payloads.
- `model_profiles.py` defines model attention geometry, portable JSON profile
  artifacts, and Qwen3 layout helpers.
- `models.py` defines cache keys, chunk references, request models, and
  materialization plans.
- `native_probe_factories.py` exposes reserved vLLM/SGLang native probe factory
  paths, delegate environment variables for backend-native adapters, and
  fail-closed backend environment diagnostics.
- `openai_compatible.py` owns the stdlib-only streaming benchmark engine for
  vLLM/SGLang OpenAI-compatible APIs.
- `planner.py` orders manifest chunks for runtime requests.
- `probe_fixtures.py` writes deterministic Qwen3 V1 adapter handoff fixtures
  for backend-native vLLM/SGLang connector development and probe debugging.
- `pr_evidence.py` emits and validates closed-schema, machine-checkable PR
  identity, traceability, Refactor-skill, and GPT-5.5 review evidence for the
  project workflow; recursive directory validation skips only clean
  validation-summary sidecars.
- `release_bundle.py` copies validated release evidence, optional benchmark
  plan execution records, Databricks run-status records, package wheels, and
  PR-evidence, release preflight, GitHub-governance, repository-hygiene, and
  native-probe factory diagnostics sidecars into a checksummed durable bundle;
  strict V1 mode requires the full release artifact set before publishing.
- `release_evidence.py` validates V1 benchmark, storage, and native engine-probe
  artifacts, including the pinned serving-engine package/version metadata.
- `repository_hygiene.py` emits a release-readiness sidecar proving `.gitignore`
  coverage and absence of tracked or untracked generated/secret-like artifacts
  that Git exposes, plus README/package-docstring coverage for repository
  directories.
- `service.py` combines planning, materialization, admission, and engine handoff.
- `serving_env.py` records pinned one-engine-per-environment install profiles
  for vLLM and SGLang helpers.
- `sglang_runtime_preflight.py` exposes the Cachet-branded CLI wrapper for the
  strict installed-SGLang HiCache dynamic-backend and provider-factory
  preflight required before provider-backed native SGLang probes.
- `sglang_smoke.py` owns the self-contained Qwen3/SGLang Databricks live smoke
  that starts a pinned SGLang server, validates the dynamic HiCache provider,
  and runs the supported baseline OpenAI-compatible live check. It fails closed
  for handoff-backed cache-arm runs until the live SGLang runtime forwards
  request metadata and SGLang page-key proof into HiCache `extra_info`.
- `storage.py` provides Memory, Disk, Unity Catalog Volume, and routed range
  readers.
- `storage_benchmark.py` measures storage-reader latency and throughput.
- `template_resources.py` lists and extracts packaged templates.
- `transformers_generator.py` provides an optional Hugging Face Transformers
  `KVChunkGenerator` implementation for model-produced prefill KV payloads.
- `vllm_smoke.py` owns the self-contained Qwen3/vLLM Databricks smoke and
  prepared-mode handoff generation/coverage validation before vLLM startup.
- `vllm_runtime_contract_data.py` single-sources the vLLM V1 KV connector
  lifecycle contract shared by native-probe diagnostics and the vendored vLLM
  adapter package.
- `vllm_runtime_preflight.py` exposes the Cachet-branded CLI wrapper for the
  strict installed-vLLM contract plus layer-name mapping preflight required
  before provider-backed native vLLM probes.
- `workflow.py` coordinates optional training, cache generation, manifest
  registration, materialization, and serving preparation.

`_reexport.py` is a private helper used by compatibility facades and is not
part of the public API.

## Compatibility-Only Modules

- `scheduler.py` remains packaged as a compatibility shim for older
  admission-helper imports. It is intentionally not advertised through
  `document_kv_cache._PUBLIC_SUBMODULES`.

## Console Scripts

The implementation package owns these document-branded CLI entry points:

- `document-kv-benchmark-plan`
- `document-kv-benchmark-handoffs`
- `document-kv-benchmark-handoff-manifest`
- `document-kv-benchmark-handoff-bundles`
- `document-kv-native-probe-scaffold`
- `document-kv-run-benchmark-plan`
- `document-kv-databricks-job`
- `document-kv-databricks-runs`
- `document-kv-storage-benchmark`
- `document-kv-storage-benchmark-databricks-job`
- `document-kv-templates`
- `document-kv-release-evidence`
- `document-kv-release-bundle`
- `document-kv-pr-evidence`
- `document-kv-github-governance`
- `document-kv-repository-hygiene`
- `document-kv-serving-env`
- `document-kv-native-probe-factories`
- `document-kv-engine-probe`
- `document-kv-engine-launch-config`
- `document-kv-engine-probe-fixture`
- `document-kv-engine-probe-databricks-job`
- `document-kv-vllm-runtime-preflight`
- `document-kv-sglang-runtime-preflight`
- `document-kv-vllm-smoke`
- `document-kv-vllm-smoke-databricks-job`
- `document-kv-sglang-smoke`
- `document-kv-sglang-smoke-databricks-job`

Cachet-branded aliases point to the same document-owned entry points:

- `cachet-benchmark-plan`
- `cachet-benchmark-handoffs`
- `cachet-benchmark-handoff-manifest`
- `cachet-benchmark-handoff-bundles`
- `cachet-native-probe-scaffold`
- `cachet-run-benchmark-plan`
- `cachet-databricks-job`
- `cachet-databricks-runs`
- `cachet-storage-benchmark`
- `cachet-storage-benchmark-databricks-job`
- `cachet-templates`
- `cachet-release-evidence`
- `cachet-release-bundle`
- `cachet-pr-evidence`
- `cachet-github-governance`
- `cachet-repository-hygiene`
- `cachet-serving-env`
- `cachet-native-probe-factories`
- `cachet-engine-probe`
- `cachet-engine-launch-config`
- `cachet-engine-probe-fixture`
- `cachet-engine-probe-databricks-job`
- `cachet-vllm-runtime-preflight`
- `cachet-sglang-runtime-preflight`
- `cachet-vllm-smoke`
- `cachet-vllm-smoke-databricks-job`
- `cachet-sglang-smoke`
- `cachet-sglang-smoke-databricks-job`

Legacy `restaurant-kv-*` aliases are no longer included in built `cachet-kv`
wheels. Use `cachet-*` or `document-kv-*` commands for installed environments.

## Optional Extras

The package keeps Databricks and test dependencies optional but exactly pinned.
Use `databricks` for managed Databricks helpers and `test` for local
verification. The vendored adapter packages are included in the Cachet wheel,
but raw vLLM and SGLang runtimes are intentionally not Poetry extras because
their current dependency graphs conflict in one resolver. Install exactly one
serving runtime in an isolated environment. `serving_env.py` exposes the pinned
helper profiles used by smoke/probe jobs.

The branded `cachet` facade, canonical `document_kv_cache` implementation, and
vendored `vllm_kv_injection`/`sglang_kv_injection` adapters ship `py.typed`
markers so downstream type checkers can consume the inline type annotations
from installed wheels.

## Template Subfolders

`templates/` contains package-data templates retrievable from installed wheels.
The `templates/databricks/` subtree mirrors the repository Databricks Asset
Bundle templates for the full V1 benchmark, standalone storage benchmark,
native engine probe, and vLLM smoke jobs.
