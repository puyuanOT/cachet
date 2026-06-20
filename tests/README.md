# Tests

These tests cover the document KV-cache orchestration package.

- `test_kvpack.py` verifies packed shard writing and local byte-range reads.
- `test_cache.py` verifies CPU hot-cache hits, local disk fallback, local budget eviction, and checksum-safe local chunk paths.
- `test_engine.py` verifies engine-ready handles, MQA/GQA layout validation, and service preparation for vLLM/SGLang adapters.
- `test_engine_adapters.py` verifies the explicit external vLLM/SGLang adapter contracts and rejection of unsupported payload or LoRA modes.
- `test_engine_probe.py` verifies the CLI runner that loads a handoff payload, invokes a backend-provided native probe factory, and emits release-gate probe records.
- `test_planner_materializer.py` verifies document planning, restaurant compatibility aliases, CPU cache reuse, and materialized byte ordering.
- `test_scheduler.py` verifies admission queue accounting for pending GPU bytes and the legacy scheduler-module compatibility shim.
- `test_storage.py` verifies Memory, Disk, Unity Catalog Volume, and routed range readers.
- `test_storage_benchmark.py` verifies the synthetic Memory/Disk/Unity Catalog reader benchmark and JSON/CLI result contract.
- `test_template_resources.py` verifies listing, reading, and extracting packaged Databricks templates.
- `test_workflow.py` verifies source-document cache generation, optional training, explicit cache-generation method labels, manifest registration, materialization, enqueue handoff, and workflow-level engine-ready handoff.
- `test_benchmarks.py` verifies V1 dataset specs, prompt/context builders, measurements, quality helpers, summary rows, baseline-vs-cache comparisons, and release-evaluable evidence checks.
- `test_dataset_prep.py` verifies raw Biography, HotpotQA, MusiQue, and NIAH conversion into canonical benchmark JSONL, including CLI and synthetic NIAH generation.
- `test_benchmark_plan.py` verifies reproducible V1 dataset-preparation plus benchmark-runner command plans for AWS g5/Qwen3 jobs.
- `test_benchmark_plan_executor.py` verifies command-plan validation, dry-run behavior, real subprocess execution, and result JSON emission.
- `test_databricks_job.py` verifies AWS g5 Databricks run-submit payload generation and runner-script rendering.
- `test_databricks_runs.py` verifies env-driven Databricks Jobs API submit/get helpers without using live credentials or network.
- `test_github_governance.py` verifies env-driven GitHub repository governance checks without using live credentials or network.
- `test_databricks_storage_benchmark_job.py` verifies AWS g5 Databricks run-submit payload generation for storage-reader evidence.
- `test_databricks_engine_probe_job.py` verifies AWS g5 Databricks run-submit payload generation for native vLLM/SGLang engine-probe evidence.
- `test_databricks_vllm_smoke_job.py` verifies AWS g5 Databricks run-submit payload generation for the self-contained vLLM smoke job.
- `test_release_evidence.py` verifies the final release gate over V1 benchmark, storage benchmark, and native vLLM/SGLang probe JSON artifacts.
- `test_release_bundle.py` verifies checksummed release-bundle packaging for validated benchmark, storage, engine-probe, release-evidence, and preflight JSON artifacts.
- `test_benchmark_runner.py` verifies canonical JSONL loading and engine-agnostic benchmark execution.
- `test_openai_compatible.py` verifies the OpenAI-compatible streaming benchmark engine with offline fake API responses.
- `test_live_server.py` verifies the optional live vLLM/SGLang smoke-check request and JSON result contract using a fake benchmark engine.
- `test_repository_hygiene.py` verifies ignore coverage and rejects tracked or
  untracked build/cache artifacts that Git exposes.

Keep tests behavioral: they should assert stable cache planning and materialization outcomes, not incidental implementation details.
