# V1 Requirements Matrix

This matrix keeps the generalized Cachet package goal auditable. Status values
mean:

- **Implemented:** source, tests, and documentation exist in the repository.
- **Bundle-refresh pending:** source/tests exist and current target evidence has
  been generated and release-validated, but the complete strict bundle still
  needs to be refreshed before publication.
- **Release-gated:** source/tests exist and current target evidence is bundled,
  but V1 publication depends on keeping the target AWS g6/L4 or Unity Catalog
  evidence fresh. Non-default g5/A10G compatibility evidence is tracked
  separately and never substitutes for the strict release target.
- **Remaining:** the repository intentionally records the work as unfinished.

## Ecosystem And Infrastructure

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Integrate with established serving platforms instead of custom solvers | Implemented | `engine_adapters.py`, `engine_probe.py`, `native_probe_factories.py`, `openai_compatible.py`, and `CONTRIBUTING.md` keep engine-specific work at the vLLM/SGLang handoff boundary. The implementation retains probes against real vLLM and SGLang native block managers, but superseded run evidence is not carried into the reset. | Run connector action descriptor validation against the frozen 0.27.1 runtime whenever connector contracts change, then keep only the refreshed native probe/action records in the strict release bundle. |
| Use Poetry with pinned dependencies | Implemented | `pyproject.toml` pins package, test, and Databricks dependencies with exact `==` requirements; `poetry.lock` records the resolver output; CI runs `poetry check --lock`. `dependency_freshness.py` and `docs/release-ops/evidence/dependency-freshness/current/dependency-freshness-evidence.json` record the current freshness policy: direct pins for `poetry-core`, `packaging`, `pyspark`, `databricks-sdk`, and `pytest` match the supplied latest stable versions; isolated vLLM/SGLang serving-profile pins are exact; fresh vLLM Q4-materializer runtime pins include `bitsandbytes` and `accelerate`; non-latest runtime holds for `sglang`, `tokenizers`, `numpy`, `fastapi`, and `prometheus-fastapi-instrumentator` carry Databricks-validation upgrade reasons; and the resolver-held `protobuf==6.33.6` drift is explained by the current `databricks-sdk==0.118.0` protobuf constraint. Strict release bundles require the matching dependency freshness sidecar. | Keep direct package pins current, refresh the dependency-freshness evidence before each release, include the dependency freshness sidecar in the strict release bundle, and rerun the relevant g6/L4 Databricks smoke or benchmark before upgrading non-latest serving-profile runtime pins. |
| Load KV ranges from Memory, Disk, and Unity Catalog | Release-gated | `storage.py`, `materializer.py`, `service.py`, and `storage_benchmark.py` cover Memory, Disk, UC Volume, and routed readers. The public storage table has been reset and contains no current numeric evidence. | Run the frozen vLLM 0.27.1 campaign for every implemented storage cell, retain backend-correlated cache-state attestation, and publish only qualified results. |
| Keep the repository clean | Implemented | `.gitignore`, `repository_hygiene.py`, directory README/docstring tests, credential scanning tests, and PR evidence validation guard generated files and secrets. | Include repository hygiene sidecar in the strict release bundle. |

## V1 Scope And Benchmarking

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Target AWS g6/L4 cluster instances | Release-gated | `databricks_job.py`, `benchmarks.py`, storage/engine/vLLM smoke job helpers, Databricks templates, and release-bundle validators consume `_hardware_targets.py`, which single-sources the `aws-g6-l4` target. The benchmark index records the frozen vLLM 0.27.1 campaign contract, but all result cells remain pending. | Qualify the final source, wheel, runtime, and inputs; complete all five independent deployment blocks; then publish a new strict bundle. |
| Restrict V1 to Qwen3 4B Instruct | Implemented | `model_profiles.py`, `vllm_smoke.py`, benchmark plans, and release evidence validate the `qwen3:4b-instruct`/`qwen3-v1` layout contract. | Re-run target evidence whenever model pins change. |
| Document quality and latency metrics | Release-gated | `benchmarks.py`, `benchmark_runner.py`, `openai_compatible.py`, and `release_evidence.py` validate TTFT, time-to-completion, throughput, answer quality, and cache-vs-baseline comparisons. The public tables are a vLLM 0.27.1-campaign-pending skeleton: implemented cells are pending and unsupported methods remain explicit `N/A`. | Complete the Baseline/Vanilla 8k/16k/32k by concurrency 1/2/4 factorial across five independent deployment blocks, all implemented ablations, and corrected full-dataset scoring before publishing numbers. |
| Benchmark Biography, HotpotQA, MusiQue, and NIAH | Release-gated | `benchmarks.py`, `dataset_prep.py`, `benchmark_plan.py`, and `vllm_smoke.py` define and smoke all four datasets. No prior run is accepted as evidence for the frozen vLLM 0.27.1 campaign. | Keep all four datasets in the strict release bundle and complete the corrected full-dataset evaluation without padding or truncating the score population. |
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
| Primary implementation language is Python | Implemented | Source packages live under `src/cachet`, `src/document_kv_cache`, and vendored `src/vllm_kv_injection`/`src/sglang_kv_injection`; the legacy restaurant facade has been removed from `src/`, and the core runtime model layer no longer retains restaurant request, service, chunk-type, key, or manifest aliases. | None for V1. |
| Provide an end-to-end API | Implemented | `workflow.py`, `service.py`, README workflow examples, `docs/native-engine-integration.md`, and tests cover optional training, cache generation, materialization, launch-config sidecars, `kv_transfer_params`, and provider-backed vLLM/SGLang handoff. | Keep native engine integration examples aligned whenever connector contracts, launch-config fields, runtime pins, or benchmark evidence change. |
| Use premium package branding | Implemented | The repository is `puyuanOT/cachet`; the distribution is the Cachet-branded `cachet-kv`; the product/import brand is Cachet with `cachet.<module>` imports plus `cachet-*` and `document-kv-*` CLI aliases. The exact `cachet` PyPI name is occupied by an unrelated Cachet API client, so `cachet-kv` is the V1 package-index identity. | Keep package metadata, README examples, and release-bundle wheel gates aligned with the Cachet brand. |
| Document every folder | Implemented | Repository governance tests require every tracked directory to have a README or package docstring. | Continue applying the directory documentation gate to every PR. |

## Workflow And Quality Gates

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| PR-driven development, no direct pushes to main | Implemented | `CONTRIBUTING.md`, `.github/main-branch-protection.json`, GitHub governance sidecars, and CI docs encode the protected-main workflow. Current GitHub governance is release-ready: the repository is public, `main` is protected with the required `Test and build` status check, branch protection applies to administrators, and unexpected open PR count is zero. | Continue using pull requests for every tracked change and keep the governance sidecar green. |
| GPT-5.5 review for each PR | Implemented | PR evidence sidecars require completed GPT-5.5 review and resolved findings. | Continue attaching PR evidence to release bundles. |
| Auto-merge approved PRs to avoid open PR buildup | Implemented | GitHub governance evidence records merge settings, auto-merge, branch deletion, and unexpected open PR counts. Current GitHub governance reports `allow_auto_merge=true`, branch deletion after merge enabled, and no unexpected open pull requests. Current operations still follow the one-PR-at-a-time merge discipline after review and green CI. | Keep exactly one PR open during active release work and merge after review plus green CI. |
| Apply Refactor skill to every PR | Implemented | PR evidence validation requires Refactor-skill evidence, and the pull request template asks reviewers to check it. | Continue recording the evidence per PR. |
| Explain what changed and why | Implemented | PR evidence schema and pull request template require `what_changed`, `why`, scope, and verification. | Continue validating PR evidence sidecars in CI and release bundles. |

## Remaining V1 Release Gates

The complete strict release bundle remains a required publication artifact.
Its closed inventory must include each governed input named by the release
validator: the release evidence sidecar, preflight sidecar, SGLang live V1
benchmark sidecar, vLLM/SGLang native engine probe sidecars, vLLM/SGLang
connector action sidecars, vLLM/SGLang engine launch config sidecars, benchmark
plan execution sidecar, Databricks run-status sidecars for benchmark, storage,
and vLLM/SGLang engine-probe runs, tested package wheel, PR evidence sidecar,
dependency freshness sidecar, V1 requirements matrix, GitHub governance
sidecar, repository hygiene sidecar, and native probe factory diagnostics
sidecar. During the 0.27.1 refresh these remain required gates, not historical
evidence that may populate the reset result tables.

- Freeze the exact vLLM 0.27.1 source, rebuilt wheels, runtime closure,
  benchmark inputs, and GPU qualification records before allocating campaign
  compute.
- Preserve the legacy-v1 `generation_throughput_with_writes` byte-identity rule
  only for historical validation; fresh native-v2 plans use a distinct
  repeat-aware sentinel requiring same-hardware fresh-load byte reproducibility
  and cross-hardware logical/token/layout/size equivalence without raw-digest
  equality.
- Use L40S as the sole publication handoff generator. Generate the reusable Q8
  handoff artifacts once on 16 independent L40S workers and bind every output
  to its qualified input and control-plane attestation.
- Run the complete Baseline/Vanilla latency factorial for 8k, 16k, and 32k at
  concurrency 1, 2, and 4 across five independent deployment blocks.
- Run every currently implemented precision, storage, hardware, and platform
  ablation plus the corrected full-dataset score evaluation. Unsupported
  methods remain explicit `N/A`.
- Publish no numeric result until the new evidence passes the canonical
  publication gate and a strict release bundle is rebuilt from the frozen
  source and wheel.
- Keep the completed legacy-facade removal contract auditable through
  `docs/legacy-compatibility-removal.md` and its generated migration evidence.
