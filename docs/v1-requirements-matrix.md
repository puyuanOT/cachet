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
| Integrate with established serving platforms instead of custom solvers | Implemented | `engine_adapters.py`, `engine_probe.py`, `native_probe_factories.py`, `openai_compatible.py`, and `CONTRIBUTING.md` keep engine-specific work at the vLLM/SGLang handoff boundary. QA run `934698284395881` completed vLLM and SGLang provider-backed native probes plus connector action descriptors against real vLLM and SGLang native block managers on `g6.8xlarge`. | Run connector action descriptor validation whenever connector contracts change, then keep the refreshed native probe/action records in the strict release bundle. |
| Use Poetry with pinned dependencies | Implemented | `pyproject.toml` pins package, test, and Databricks dependencies with exact `==` requirements; `poetry.lock` records the resolver output; CI runs `poetry check --lock`. `dependency_freshness.py` and `docs/release-ops/evidence/dependency-freshness/current/dependency-freshness-evidence.json` record the current freshness policy: direct pins for `poetry-core`, `packaging`, `pyspark`, `databricks-sdk`, and `pytest` match the supplied latest stable versions; isolated vLLM/SGLang serving-profile pins are exact; fresh vLLM Q4-materializer runtime pins include `bitsandbytes` and `accelerate`; non-latest runtime holds for `sglang`, `tokenizers`, `numpy`, `fastapi`, and `prometheus-fastapi-instrumentator` carry Databricks-validation upgrade reasons; and the resolver-held `protobuf==6.33.6` drift is explained by the current `databricks-sdk==0.118.0` protobuf constraint. Strict release bundles require the matching dependency freshness sidecar. | Keep direct package pins current, refresh the dependency-freshness evidence before each release, include the dependency freshness sidecar in the strict release bundle, and rerun the relevant g6/L4 Databricks smoke or benchmark before upgrading non-latest serving-profile runtime pins. |
| Load KV ranges from Memory, Disk, and Unity Catalog | Implemented | `storage.py`, `materializer.py`, `service.py`, and `storage_benchmark.py` cover Memory, Disk, UC Volume, and routed readers. The current Q4/Q8 Vanilla serving comparison records 256 requests each for a prewarmed 16 GiB RAM payload cache, cold-attested local disk, and a mounted Unity Catalog path. RAM records 256/256 measured payload-cache hits and zero backend reads. Unity Catalog OS page-cache eviction succeeded, but its backend cache state is unproven, so it is not labeled strict cold-UC evidence. Historical reader-only evidence remains available in release provenance. | Re-run storage evidence whenever storage readers, UC access patterns, or bundle schema gates change; add backend-correlated UC cache-state attestation before making a strict cold-UC claim. |
| Keep the repository clean | Implemented | `.gitignore`, `repository_hygiene.py`, directory README/docstring tests, credential scanning tests, and PR evidence validation guard generated files and secrets. | Include repository hygiene sidecar in the strict release bundle. |

## V1 Scope And Benchmarking

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Target AWS g6/L4 cluster instances | Release-gated | `databricks_job.py`, `benchmarks.py`, storage/engine/vLLM smoke job helpers, Databricks templates, and release-bundle validators consume `_hardware_targets.py`, which single-sources the default `aws-g6-l4` benchmark id, default `g6.8xlarge` node, and `g6.` Databricks node-family policy while also allowing the explicit non-default `aws-g5-a10g`/`g5.` compatibility target. `benchmarks/README.md` is the research-style human-facing benchmark index for the current Qwen3-4B 4-bit-weight + Q8-document-KV protocol. Its g6/L4 request-parallelism-4 cold-hydrate table and the matched g5/A10G 16k setting comparison are populated from compact sanitized descriptive evidence. The node types have different local-disk topologies (two 450 GB disks versus one 900 GB disk), so storage topology can contribute to the hardware comparison. The isolated Baseline raw record does not pass the generic canary gate, and no current result is publication-qualified. Historical A10G warm-prefix and BF16 canary evidence remain separate under `benchmarks/appendix/`. | Pass a canonical canary and publication gate, then refresh g6/L4 and g5/A10G evidence whenever benchmark, model, native connector contracts, package wheel identity, PR evidence, dependency freshness, or the current appendix benchmark folder changes. |
| Restrict V1 to Qwen3 4B Instruct | Implemented | `model_profiles.py`, `vllm_smoke.py`, benchmark plans, and release evidence validate the `qwen3:4b-instruct`/`qwen3-v1` layout contract. | Re-run target evidence whenever model pins change. |
| Document quality and latency metrics | Release-gated | `benchmarks.py`, `benchmark_runner.py`, `openai_compatible.py`, and `release_evidence.py` validate TTFT, time-to-completion, throughput, answer quality, and cache-vs-baseline comparisons. The public benchmark folder reports current Qwen3-4B 4-bit-weight + Q8-document-KV Baseline and Vanilla latency through 32k plus coupled precision, RAM/disk/mounted-UC storage, g6/L4-versus-g5/A10G, and vLLM serving tables. Every measured latency/resource job uses two examples per dataset, 32 repeats per example, request parallelism 4, and 256 successful requests. These rows are descriptive/nonpublication-qualified because the isolated Baseline raw record fails the generic cache-arm/resource-schema canary gate. Full-dataset score cells are explicit `N/A`; a separate matched five-example-per-dataset score diagnostic is not promoted to a full-dataset or superiority claim. Packed-Q4, Q8 pre-RoPE SGLang, KV Packet, CacheBlend, InfoFlow KV, LongBench v2, and RULER remain unsupported or unimplemented and are explicit `N/A`. | Pass a canonical canary and publication gate over meaningful distinct-sample coverage with example-clustered uncertainty and approved scorers. Refresh the compact appendix evidence whenever benchmark code, runtime pins, connector behavior, package wheel identity, serving platform, or precision changes. Cold-hydrate rows use per-request `cache_salt` isolation plus OS page-cache eviction; mounted Unity Catalog backend-cache state remains unproven. |
| Benchmark Biography, HotpotQA, MusiQue, and NIAH | Release-gated | `benchmarks.py`, `dataset_prep.py`, `benchmark_plan.py`, and `vllm_smoke.py` define and smoke all four datasets. Historical QA benchmark run `872615985402004` completed those datasets with a then-valid release bundle, but its post-RoPE benchmark record is superseded and does not satisfy the current Vanilla protocol. | Keep all four datasets in every strict V1 release bundle and complete a refreshed final-wheel main-protocol run when benchmark code, model pins, native connector behavior, or package wheel identity changes. |
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

- Current Q4-weight/Q8-KV latency and ablation tables are populated from compact
  sanitized evidence over two examples per dataset, 32 repeats per example,
  request parallelism 4, and 256 requests per isolated job. The evidence is
  descriptive/nonpublication-qualified: the isolated Baseline raw record fails
  the generic canary gate because it has no cache arm and run-level resource
  telemetry is outside the per-arm resource schema. Full-dataset scores remain
  `N/A`; the separate five-example-per-dataset diagnostic supports no
  full-dataset or superiority claim. A canonical canary and publication gate
  over meaningful distinct-sample coverage remain release gates.
- Historical, superseded post-RoPE g6/L4 benchmark evidence exists for
  `cachet_vllm_hot_payload_longcmp_388ea0a_20260623_160711_repeat3_cache8g_cachet_kv_current_main`
  from QA Databricks run `872615985402004` on a single-node `g6.8xlarge`: all
  four datasets completed with no benchmark errors and 24 measurements.
  The installed package was `cachet_kv-0.2.0-py3-none-any.whl`, the vLLM import
  probe reported `DocumentKVNativeProvider`, and the vLLM server log recorded
  external prefix-cache hits plus successful Cachet layer loads
  (`document_kv_layers_loaded=36`, `document_kv_load_error_blocks=0`). This
  record remains useful for release-ops provenance, but its post-RoPE method and
  unreconstructable inputs cannot support a current Vanilla latency or
  quality claim.
- Historical, superseded post-RoPE g5/A10G compatibility benchmark evidence
  exists for
  `cachet_vllm_hot_payload_g5_longcmp_388ea0a_20260623_162302_repeat3_cache8g_cachet_kv_current_main`
  from QA Databricks run `566743786103032` on a single-node `g5.8xlarge`: all
  four datasets completed with no benchmark errors and 24 measurements. Release
  evidence over that historical g5 benchmark plus the then-current storage and native
  vLLM/SGLang probe/action artifacts is `ok=true` with no issues. The installed
  package was `cachet_kv-0.2.0-py3-none-any.whl`, and the vLLM server log
  recorded native `DocumentKVConnector` startup, payload-cache hits, successful
  Cachet layer loads (`document_kv_layers_loaded=36`), and zero load error
  blocks (`document_kv_load_error_blocks=0`). This compatibility evidence can
  be bundled through the optional `compatibility_benchmark` artifact role and
  does not replace the strict V1 publication target, which remains the default
  AWS g6/L4 release bundle. It also does not support a current Vanilla
  performance or quality claim.
- Target g6/L4 UC storage-reader evidence exists for
  `cachet_readiness_20260621_095026` from QA Databricks run
  `948365719597221`: Memory, Disk, and Unity Catalog readers all completed with
  zero errors against a real UC Volume. Its Databricks run-status sidecar was
  regenerated with the current strict bundle schema, including explicit
  `spark_env_keys` arrays.
- Target g6/L4 native engine evidence exists for QA Databricks run
  `934698284395881`: vLLM and SGLang provider-backed native probe tasks both
  terminated `SUCCESS` on `g6.8xlarge`, emitted `payload_mode=merged` engine
  probe sidecars, connector action sidecars, runtime preflight sidecars, and
  native probe factory diagnostics from inside the installed runtime
  environments. Run connector action descriptor validation remains the required
  regression step whenever connector contracts change.
- Historical release-evidence validation over the then-current target benchmark,
  storage, and vLLM/SGLang native probe/action artifacts was `ok=true` with no
  issues; the same was true for the historical `aws-g5-a10g` compatibility
  benchmark through the `compatibility_benchmark` role. That superseded
  strict-bundle snapshot was built after PR #513 with its then-current wheel
  and validated 37 artifacts, including the historical
  `872615985402004`/`566743786103032` benchmark/status evidence, the SGLang live
  V1 sidecar from run `48413356233422`,
  PR #442/#503/#504/#505/#506/#507/#508/#509/#510/#511/#512/#513 evidence,
  carrying `legacy_migration_evidence` for the removed restaurant facade, and
  carrying `dependency_freshness` evidence for the current package/runtime
  dependency policy.
  It remains an audit snapshot, not evidence for the current source, wheel, or
  Vanilla main protocol. A refreshed complete strict release bundle built from
  the final wheel and a canonical publication-qualified rerun remain pending.
  Traceability-only PR evidence
  added after that snapshot must be included in that refresh. The historical
  bundled artifacts include the release evidence
  sidecar, preflight sidecar, vLLM/SGLang native engine probe sidecars,
  vLLM/SGLang connector action sidecars, vLLM/SGLang engine launch config
  sidecars, SGLang live V1 benchmark sidecar, benchmark plan execution sidecar,
  Databricks run-status sidecars for benchmark, storage, and vLLM/SGLang engine-probe runs, the
  `aws-g5-a10g` compatibility benchmark, the
  `compatibility_databricks_run_status` sidecar, tested package wheel, PR
  evidence sidecar, legacy migration evidence
  sidecar, dependency freshness sidecar, V1 requirements matrix, GitHub
  governance sidecar, repository hygiene sidecar, and native probe factory
  diagnostics sidecars emitted from the split vLLM/SGLang runtime probe
  environments.
- GitHub governance is release-ready: the repository is public, visibility is
  public, auto-merge is enabled, merged head branches are deleted automatically,
  `main` branch protection requires the `Test and build` check, branch
  protection applies to administrators, force-pushes/deletions are blocked, and
  no unexpected pull requests are open.
- `benchmark_plan.py` can now emit the vLLM/SGLang engine launch config sidecars
  through `--engine-launch-config-output-dir`; those generated paths satisfy the
  strict bundle launch-config gate when paired with the native probe/action
  evidence.
- Keep runtime serving inside established engines and outside Cachet's package
  boundary.
- The built `cachet-kv` wheel and source tree no longer include the legacy
  restaurant facade, `restaurant-kv-*` scripts, or core runtime aliases for
  restaurant request models, chunk types, key fields, manifest lookup helpers,
  and service names. Current downstream migration evidence is tracked under
  `docs/release-ops/evidence/legacy-migration/current/` as validated
  `document_kv.legacy_compatibility_migration.v1` evidence that can be bundled
  through the optional `legacy_migration_evidence` role; see
  `docs/legacy-compatibility-removal.md` for the completed removal contract.
