# V1 Requirements Matrix

This matrix keeps the generalized Cachet package goal auditable. Status values
mean:

- **Implemented:** source, tests, and documentation exist in the repository.
- **Release-gated:** source/tests exist and current target evidence is bundled,
  but V1 publication depends on keeping the target AWS g6/L4 or Unity Catalog
  evidence fresh. Non-default g5/A10G compatibility evidence is tracked
  separately and never substitutes for the strict release target.
- **Remaining:** the repository intentionally records the work as unfinished.

## Ecosystem And Infrastructure

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Integrate with established serving platforms instead of custom solvers | Implemented | `engine_adapters.py`, `engine_probe.py`, `native_probe_factories.py`, `openai_compatible.py`, and `CONTRIBUTING.md` keep engine-specific work at the vLLM/SGLang handoff boundary. The implementation retains probes against real vLLM and SGLang native block managers. | Run connector action descriptor validation against the pinned vLLM 0.30.0 runtime whenever connector contracts change, then keep only the refreshed native probe/action records in the strict release bundle. |
| Use Poetry with pinned dependencies | Implemented | `pyproject.toml` pins package, test, and Databricks dependencies with exact `==` requirements; `poetry.lock` records the resolver output; CI runs `poetry check --lock`. `dependency_freshness.py` and `docs/release-ops/evidence/dependency-freshness/current/dependency-freshness-evidence.json` record the current freshness policy: direct pins for `poetry-core`, `packaging`, `databricks-sdk`, and `pytest` match the supplied latest stable versions; isolated vLLM/SGLang serving-profile pins are exact; fresh vLLM Q4-materializer runtime pins include `bitsandbytes` and `accelerate`; non-latest runtime holds for `sglang`, `tokenizers`, `numpy`, `fastapi`, and `prometheus-fastapi-instrumentator` carry Databricks-validation upgrade reasons; and the resolver-held `protobuf==6.33.6` drift is explained by the current `databricks-sdk==0.118.0` protobuf constraint. Strict release bundles require the matching dependency freshness sidecar. | Keep direct package pins current, refresh the dependency-freshness evidence before each release, include the dependency freshness sidecar in the strict release bundle, and rerun the relevant g6/L4 Databricks smoke or benchmark before upgrading non-latest serving-profile runtime pins. |
| Load KV ranges from Memory, Disk, and Unity Catalog | Release-gated | `storage.py`, `materializer.py`, `service.py`, and `storage_benchmark.py` cover Memory, Disk, UC Volume, and routed readers. The published vLLM 0.30.0 campaign measured local NVMe disk, RAM, and the Unity Catalog mounted path on L4, and the auxiliary latency re-run reproduces those storage contrasts; the hybrid RAM/disk/Unity Catalog serving policy is not implemented. | Keep backend-correlated cache-state attestation in future storage runs, publish only gate-qualified results, and include the storage evidence in the next strict release bundle. |
| Keep the repository clean | Implemented | `.gitignore`, `repository_hygiene.py`, directory README/docstring tests, and credential scanning tests guard generated files and secrets. | Include repository hygiene sidecar in the strict release bundle. |

## V1 Scope And Benchmarking

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Target AWS g6/L4 cluster instances | Release-gated | `databricks_job.py`, `benchmarks.py`, storage/engine/vLLM smoke job helpers, Databricks templates, and release-bundle validators consume `_hardware_targets.py`, which single-sources the `aws-g6-l4` target. The published vLLM 0.30.0 campaign ran its core serving on AWS g6/L4. | Rebuild the strict release bundle from source-bound g6/L4 artifacts before publication. |
| Restrict V1 to Qwen3 4B Instruct | Implemented | `model_profiles.py`, `vllm_smoke.py`, benchmark plans, and release evidence validate the `qwen3:4b-instruct`/`qwen3-v1` layout contract. | Re-run target evidence whenever model pins change. |
| Document quality and latency metrics | Release-gated | `benchmarks.py`, `benchmark_runner.py`, `openai_compatible.py`, and `release_evidence.py` validate TTFT, time-to-completion, throughput, answer quality, and cache-vs-baseline comparisons; release evidence re-scores every measurement with its dataset's Quality v2 scorer. The public tables are rendered from the sealed vLLM 0.30.0 campaign records: implemented cells carry published values and unsupported methods remain explicit `N/A`. | Publish new numbers only from sealed records that pass the publication gate. |
| Benchmark Biography, HotpotQA, MusiQue, and NIAH | Release-gated | `benchmarks.py`, `dataset_prep.py`, `benchmark_plan.py`, and `vllm_smoke.py` define and smoke all four datasets, prompted and scored by `quality_v2.py`, and `benchmark_dataset_sources.py` stages NIAH as the Quality v2 natural-paragraph grid. Quality Benchmark v2 publishes answer quality for every selected HotpotQA, MusiQue, and NIAH row and a fixed 2,000-example Biography diagnostic subset. | Keep all four datasets in the strict release bundle and score full datasets without padding or truncation. |
| Compare against standard no-cache prefill | Implemented | Benchmark summaries require a `baseline_prefill` arm and cache-arm comparisons with logical/runtime prompt accounting. | Target release evidence must include finite baseline and cache measurements. |

## Architecture And Extensibility

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Support MQA/GQA K/V layout metadata | Implemented | `model_profiles.py`, `engine_protocol.py`, and release evidence validate shared K/V storage, bytes-per-token, and stride geometry. | Add new profile records when future model families become release targets. |
| Support hot CPU and cold disk cache tiers | Implemented | `cache.py`, `service.py`, and README examples expose CPU LRU plus local disk tiering. | Capture workload-specific sizing in deployment docs once production traffic is known. |
| Leave room for KV Packet or adapter-trained methods | Implemented | `workflow.py` models optional training adapters, cache generation methods, adapter artifacts, and engine adapter IDs. | Add real training-backed integrations outside V1's vanilla cache path. |
| Scale to Qwen3.5 and MiniMax-style future models | Implemented | `ModelProfileRegistry` supports caller-owned model profiles and docs/tests cover future GQA/MQA-style profile extension. | Promote future model profiles only after validated engine and benchmark evidence exists. |

## Usability, Branding, And Documentation

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Primary implementation language is Python | Implemented | Source packages live under `src/cachet`, `src/document_kv_cache`, and vendored `src/vllm_kv_injection`/`src/sglang_kv_injection`, and the built `cachet-kv` wheel contains exactly those four packages. | None for V1. |
| Provide an end-to-end API | Implemented | `workflow.py`, `service.py`, README workflow examples, `docs/native-engine-integration.md`, and tests cover optional training, cache generation, materialization, launch-config sidecars, `kv_transfer_params`, and provider-backed vLLM/SGLang handoff. | Keep native engine integration examples aligned whenever connector contracts, launch-config fields, runtime pins, or benchmark evidence change. |
| Use premium package branding | Implemented | The repository is `puyuanOT/cachet`; the distribution is the Cachet-branded `cachet-kv`; the product/import brand is Cachet with `cachet.<module>` imports and `cachet-*` CLI commands. The exact `cachet` PyPI name is occupied by an unrelated Cachet API client, so `cachet-kv` is the V1 package-index identity. | Keep package metadata, README examples, and release-bundle wheel gates aligned with the Cachet brand. |
| Document every folder | Implemented | Repository governance tests require every tracked directory to have a README or package docstring. | Continue applying the directory documentation gate to every PR. |

## Workflow And Quality Gates

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| PR-driven development, no direct pushes to main | Implemented | `CONTRIBUTING.md`, `.github/main-branch-protection.json`, GitHub governance sidecars, and CI docs encode the protected-main workflow. Current GitHub governance is release-ready: the repository is public, `main` is protected with the required `Test and build` status check, branch protection applies to administrators, and unexpected open PR count is zero. | Continue using pull requests for every tracked change and keep the governance sidecar green. |
| GPT-5.5 review for each PR | Implemented | `cachet-pr-evidence` rejects a sidecar without a completed GPT-5.5 review whose outcome is clean or findings-resolved, and the strict release bundle requires the release PR's sidecar. CI does not check other PRs. | Generate the release PR's sidecar at bundle time. |
| Auto-merge approved PRs to avoid open PR buildup | Implemented | GitHub governance evidence records merge settings, auto-merge, branch deletion, and unexpected open PR counts. Current GitHub governance reports `allow_auto_merge=true`, branch deletion after merge enabled, and no unexpected open pull requests. Current operations still follow the one-PR-at-a-time merge discipline after review and green CI. | Keep exactly one PR open during active release work and merge after review plus green CI. |
| Apply Refactor skill to every PR | Implemented | `cachet-pr-evidence` rejects a sidecar without `--refactor-skill-applied`, and the strict release bundle requires the release PR's sidecar. CI and the pull request template do not check other PRs. | Record it in the release PR's sidecar at bundle time. |
| Explain what changed and why | Implemented | The pull request template has What Changed, Why, Scope, and Verification sections, and `cachet-pr-evidence` requires `what_changed`, `why`, scope, and verification entries. | Fill in the template for every PR and include the release PR's sidecar in the strict release bundle. |

## Remaining V1 Release Gates

The complete strict release bundle is the outstanding publication artifact.
Build it from source-bound artifacts. Its closed inventory must include each
governed input named by the release validator: the release evidence sidecar, preflight sidecar, benchmark
evidence gate sidecar, SGLang live V1 benchmark sidecar, vLLM/SGLang native engine probe sidecars, vLLM/SGLang
connector action sidecars, vLLM/SGLang engine launch config sidecars, benchmark
plan execution sidecar, Databricks run-status sidecars for benchmark, storage,
and vLLM/SGLang engine-probe runs, tested package wheel, PR evidence sidecar,
dependency freshness sidecar, V1 requirements matrix, GitHub governance
sidecar, repository hygiene sidecar, and native probe factory diagnostics
sidecar. The sealed benchmark records in `benchmarks/` are not bundle inputs;
`benchmarks/verify_sealed_records.py` checks them.
