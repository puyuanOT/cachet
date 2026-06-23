# V1 Requirements Matrix

This matrix keeps the generalized Cachet package goal auditable. Status values
mean:

- **Implemented:** source, tests, and documentation exist in the repository.
- **Release-gated:** source/tests exist, but V1 publication still needs target
  AWS g6/L4 or Unity Catalog evidence in the release bundle. Non-default
  g5/A10G compatibility evidence is tracked separately and never substitutes
  for the strict release target.
- **Remaining:** the repository intentionally records the work as unfinished.

## Ecosystem And Infrastructure

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Integrate with established serving platforms instead of custom solvers | Implemented | `engine_adapters.py`, `engine_probe.py`, `native_probe_factories.py`, `openai_compatible.py`, and `CONTRIBUTING.md` keep engine-specific work at the vLLM/SGLang handoff boundary. QA run `934698284395881` completed vLLM and SGLang provider-backed native probes plus connector action descriptors against real vLLM and SGLang native block managers on `g6.8xlarge`. | Run connector action descriptor validation whenever connector contracts change, then keep the refreshed native probe/action records in the strict release bundle. |
| Use Poetry with pinned dependencies | Implemented | `pyproject.toml` pins package, test, Databricks, and helper environment dependencies; CI runs `poetry check`. | Keep pins current before each release. |
| Load KV ranges from Memory, Disk, and Unity Catalog | Implemented | `storage.py`, `materializer.py`, `service.py`, and `storage_benchmark.py` cover Memory, Disk, UC Volume, and routed readers. QA run `948365719597221` produced Memory, Disk, and real Unity Catalog storage-reader evidence; its run-status sidecar has been regenerated with current strict bundle schema. | Include the refreshed storage benchmark JSON and Databricks run-status sidecar in the strict release bundle after GitHub governance is release-ready. |
| Keep the repository clean | Implemented | `.gitignore`, `repository_hygiene.py`, directory README/docstring tests, credential scanning tests, and PR evidence validation guard generated files and secrets. | Include repository hygiene sidecar in the strict release bundle. |

## V1 Scope And Benchmarking

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Target AWS g6/L4 cluster instances | Release-gated | `databricks_job.py`, `benchmarks.py`, storage/engine/vLLM smoke job helpers, Databricks templates, and release-bundle validators consume `_hardware_targets.py`, which single-sources the default `aws-g6-l4` benchmark id, default `g6.8xlarge` node, and `g6.` Databricks node-family policy while also allowing the explicit non-default `aws-g5-a10g`/`g5.` compatibility target. Successful QA Databricks status sidecars now exist for the benchmark run `426398182137665`, storage run `948365719597221`, and native engine-probe run `934698284395881` on the default AWS g6/L4 target; current g5/A10G compatibility benchmark run `315109189523858` completed on `g5.8xlarge` with release evidence `ok=true` when paired with the current storage and native engine sidecars. Release bundles can carry that g5 report as a `compatibility_benchmark` artifact without letting it substitute for the strict release target. | Publish the complete strict g6/L4 release bundle once GitHub governance is release-ready, and keep g5/A10G compatibility evidence refreshed when benchmark, model, or native connector contracts change. |
| Restrict V1 to Qwen3 4B Instruct | Implemented | `model_profiles.py`, `vllm_smoke.py`, benchmark plans, and release evidence validate the `qwen3:4b-instruct`/`qwen3-v1` layout contract. | Re-run target evidence whenever model pins change. |
| Document quality and latency metrics | Release-gated | `benchmarks.py`, `benchmark_runner.py`, `openai_compatible.py`, and `release_evidence.py` validate TTFT, time-to-completion, throughput, answer quality, and cache-vs-baseline comparisons. QA benchmark run `426398182137665` produced `document_kv.benchmark_run.v1` evidence with 24 measurements, 4 comparisons, zero quality deltas, TTFT speedups from 5.18x to 6.78x, and time-to-completion speedups from 1.74x to 2.22x. | Bundle the refreshed benchmark report and run-status sidecar in the strict release bundle after GitHub governance is release-ready. |
| Benchmark Biography, HotpotQA, MusiQue, and NIAH | Release-gated | `benchmarks.py`, `dataset_prep.py`, `benchmark_plan.py`, and `vllm_smoke.py` define and smoke all four datasets. QA benchmark run `426398182137665` completed Biography, HotpotQA, MusiQue, and NIAH with release evidence `ok=true`. | Keep all four datasets in every strict V1 release bundle and re-run when benchmark code, model pins, or native connector behavior changes. |
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
| Primary implementation language is Python | Implemented | Source packages live under `src/cachet`, `src/document_kv_cache`, vendored `src/vllm_kv_injection`/`src/sglang_kv_injection`, and the legacy compatibility facade. | None for V1. |
| Provide an end-to-end API | Implemented | `workflow.py`, `service.py`, README workflow examples, and tests cover optional training, cache generation, materialization, and engine handoff. | Add native engine integration examples after connector probes land. |
| Use premium package branding | Implemented | The repository is `puyuanOT/cachet`; the distribution remains the transitional `document-kv-cache`; the product/import brand is Cachet with `cachet.<module>` imports plus `cachet-*` and `document-kv-*` CLI aliases. | Complete the package-index migration before public publication. |
| Document every folder | Implemented | Repository governance tests require every tracked directory to have a README or package docstring. | Continue applying the directory documentation gate to every PR. |

## Workflow And Quality Gates

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| PR-driven development, no direct pushes to main | Release-gated | `CONTRIBUTING.md`, `.github/main-branch-protection.json`, GitHub governance sidecars, and CI docs encode the protected-main workflow. | GitHub currently reports the repository as private and main branch protection disabled; make the repository public, then apply the branch-protection policy before public release. |
| GPT-5.5 review for each PR | Implemented | PR evidence sidecars require completed GPT-5.5 review and resolved findings. | Continue attaching PR evidence to release bundles. |
| Auto-merge approved PRs to avoid open PR buildup | Release-gated | GitHub governance evidence records merge settings, auto-merge, branch deletion, and unexpected open PR counts. Current operations still follow the one-PR-at-a-time merge discipline after review and green CI. | GitHub currently reports `allow_auto_merge=false`; enable auto-merge before public release, keep exactly one PR open during active release work, and merge after review plus green CI. |
| Apply Refactor skill to every PR | Implemented | PR evidence validation requires Refactor-skill evidence, and the pull request template asks reviewers to check it. | Continue recording the evidence per PR. |
| Explain what changed and why | Implemented | PR evidence schema and pull request template require `what_changed`, `why`, scope, and verification. | Continue validating PR evidence sidecars in CI and release bundles. |

## Remaining V1 Release Gates

- Target g6/L4 benchmark evidence exists for
  `cachet_vllm_hot_payload_9ec0657_20260623_053557_repeat3_cache8g_current_main`
  from QA Databricks run `426398182137665` on a single-node `g6.8xlarge`: all
  four datasets completed with no benchmark errors, 24 measurements, 4
  cache-vs-baseline comparisons, answer-found and exact-match deltas of zero,
  TTFT speedups of 5.18x-6.78x, and time-to-completion speedups of 1.74x-2.22x.
  The vLLM server log recorded external prefix-cache hits and successful
  Cachet layer loads (`document_kv_layers_loaded=36`,
  `document_kv_load_error_blocks=0`).
- Current g5/A10G compatibility benchmark evidence exists for
  `cachet_vllm_hot_payload_g5_01a6147_20260623_125720_repeat3_cache8g_current_main`
  from QA Databricks run `315109189523858` on a single-node `g5.8xlarge`: all
  four datasets completed with no benchmark errors, 24 measurements, 4
  cache-vs-baseline comparisons, `v1_evidence.ok=true`, TTFT speedups of
  4.55x-6.07x, and time-to-completion speedups of 2.03x-2.68x. Release
  evidence over that g5 benchmark plus the current storage and native
  vLLM/SGLang probe/action artifacts is `ok=true` with no issues. The vLLM
  server log recorded native `DocumentKVConnector` startup, payload-cache hits,
  successful Cachet layer loads (`document_kv_layers_loaded=36`), and zero load
  error blocks (`document_kv_load_error_blocks=0`). This compatibility evidence
  can be bundled through the optional `compatibility_benchmark` artifact role
  and does not replace the strict V1 publication target, which remains the
  default AWS g6/L4 release bundle.
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
- Release-evidence validation over the target benchmark, storage, and fresh
  vLLM/SGLang native probe/action artifacts is `ok=true` with no issues. An
  enriched release bundle without GitHub governance built successfully with 20
  artifacts. A g5-enriched no-governance bundle also validates with
  21 artifacts when the current `aws-g5-a10g` compatibility benchmark is added
  via the `compatibility_benchmark` role. The bundled artifacts include the release
  evidence sidecar, preflight sidecar, vLLM/SGLang native engine probe sidecars,
  vLLM/SGLang connector action sidecars, vLLM/SGLang engine launch config
  sidecars, benchmark plan execution sidecar, Databricks run-status sidecars
  for benchmark, storage, and vLLM/SGLang engine-probe runs, tested package
  wheel, PR evidence sidecar, V1 requirements matrix, repository hygiene
  sidecar, and native probe factory diagnostics sidecars emitted from the split
  vLLM/SGLang runtime probe environments.
- The complete strict release bundle still requires the GitHub governance
  sidecar to be release-ready. Current GitHub governance reports the repository
  as private, repository visibility as private, `allow_auto_merge=false`, and
  main branch protection disabled; the branch-protection API returns a 403 that
  says to upgrade GitHub Pro or make the repository public. Make the repository
  public, enable auto-merge, apply the main branch-protection policy,
  regenerate the GitHub governance sidecar, and then rebuild the complete
  strict release bundle with the full strict artifact set.
- `benchmark_plan.py` can now emit the vLLM/SGLang engine launch config sidecars
  through `--engine-launch-config-output-dir`; those generated paths satisfy the
  strict bundle launch-config gate when paired with the native probe/action
  evidence.
- Keep runtime serving inside established engines and outside Cachet's package
  boundary.
- Remove the `restaurant_kv_serving` compatibility package only after downstream
  jobs migrate.
