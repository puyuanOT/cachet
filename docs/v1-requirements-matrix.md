# V1 Requirements Matrix

This matrix keeps the generalized Cachet package goal auditable. Status values
mean:

- **Implemented:** source, tests, and documentation exist in the repository.
- **Release-gated:** source/tests exist, but V1 publication still needs target
  AWS g5 or Unity Catalog evidence in the release bundle.
- **Remaining:** the repository intentionally records the work as unfinished.

## Ecosystem And Infrastructure

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Integrate with established serving platforms instead of custom solvers | Implemented | `engine_adapters.py`, `engine_probe.py`, `native_probe_factories.py`, `openai_compatible.py`, and `CONTRIBUTING.md` keep engine-specific work at the vLLM/SGLang handoff boundary. | Run native connector action probes against real vLLM and SGLang block managers. |
| Use Poetry with pinned dependencies | Implemented | `pyproject.toml` pins package, test, Databricks, and helper environment dependencies; CI runs `poetry check`. | Keep pins current before each release. |
| Load KV ranges from Memory, Disk, and Unity Catalog | Implemented | `storage.py`, `materializer.py`, `service.py`, and `storage_benchmark.py` cover Memory, Disk, UC Volume, and routed readers. | Publish UC-backed storage benchmark evidence from the target workspace. |
| Keep the repository clean | Implemented | `.gitignore`, `repository_hygiene.py`, directory README/docstring tests, credential scanning tests, and PR evidence validation guard generated files and secrets. | Include repository hygiene sidecar in the strict release bundle. |

## V1 Scope And Benchmarking

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| Target AWS g5 cluster instances | Release-gated | `databricks_job.py`, storage/engine/vLLM smoke job helpers, Databricks templates, and release-bundle validators require AWS g5 node summaries. | Run and attach terminal successful Databricks status sidecars for benchmark, storage, and engine-probe jobs. |
| Restrict V1 to Qwen3 4B Instruct | Implemented | `model_profiles.py`, `vllm_smoke.py`, benchmark plans, and release evidence validate the `qwen3:4b-instruct`/`qwen3-v1` layout contract. | Re-run target evidence whenever model pins change. |
| Document quality and latency metrics | Release-gated | `benchmarks.py`, `benchmark_runner.py`, `openai_compatible.py`, and `release_evidence.py` validate TTFT, time-to-completion, throughput, answer quality, and cache-vs-baseline comparisons. | Publish complete V1 benchmark reports from target AWS g5 runs. |
| Benchmark Biography, HotpotQA, MusiQue, and NIAH | Release-gated | `benchmarks.py`, `dataset_prep.py`, `benchmark_plan.py`, and `vllm_smoke.py` define and smoke all four datasets. | Run the full dataset plan and bundle the resulting release evidence. |
| Compare against standard no-cache prefill | Implemented | Benchmark summaries require a `full_no_cache` baseline and cache-arm comparisons with logical/runtime prompt accounting. | Target release evidence must include finite baseline and cache measurements. |

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
| Primary implementation language is Python | Implemented | Source packages live under `src/cachet`, `src/document_kv_cache`, and the legacy compatibility facade. | None for V1. |
| Provide an end-to-end API | Implemented | `workflow.py`, `service.py`, README workflow examples, and tests cover optional training, cache generation, materialization, and engine handoff. | Add native engine integration examples after connector probes land. |
| Use premium package branding | Implemented | The distribution is `document-kv-cache`; the product/import brand is Cachet with `cachet-*` and `document-kv-*` CLI aliases. | Keep repository metadata synchronized before publishing. |
| Document every folder | Implemented | Repository governance tests require every tracked directory to have a README or package docstring. | Continue applying the directory documentation gate to every PR. |

## Workflow And Quality Gates

| Requirement | Status | Current Evidence | Remaining Gate |
| --- | --- | --- | --- |
| PR-driven development, no direct pushes to main | Implemented | `CONTRIBUTING.md`, `.github/main-branch-protection.json`, GitHub governance sidecars, and CI docs encode the protected-main workflow. | Maintainers must keep branch protection applied in GitHub. |
| GPT-5.5 review for each PR | Implemented | PR evidence sidecars require completed GPT-5.5 review and resolved findings. | Continue attaching PR evidence to release bundles. |
| Auto-merge approved PRs to avoid open PR buildup | Implemented | GitHub governance evidence records merge settings, auto-merge, branch deletion, and unexpected open PR counts. | Keep operational merge discipline: one PR open, merge after review and green CI. |
| Apply Refactor skill to every PR | Implemented | PR evidence validation requires Refactor-skill evidence, and the pull request template asks reviewers to check it. | Continue recording the evidence per PR. |
| Explain what changed and why | Implemented | PR evidence schema and pull request template require `what_changed`, `why`, scope, and verification. | Continue validating PR evidence sidecars in CI and release bundles. |

## Remaining V1 Release Gates

- Run and publish the complete strict release bundle from target AWS g5/Unity
  Catalog runs with the full strict artifact set: release evidence sidecar,
  preflight sidecar, vLLM/SGLang native engine probe sidecars, vLLM/SGLang
  connector action sidecars, benchmark plan execution sidecar, Databricks
  run-status sidecar, tested package wheel, PR evidence sidecar, V1 requirements
  matrix, GitHub governance sidecar, repository hygiene sidecar, and native
  probe factory diagnostics sidecar.
- Run connector action descriptor validation against real vLLM and SGLang native block managers.
- Keep runtime serving inside established engines and outside Cachet's package
  boundary.
- Remove the `restaurant_kv_serving` compatibility package only after downstream
  jobs migrate.
