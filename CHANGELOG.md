# Changelog

## 0.3.0

- Direct pins refreshed to `poetry-core==2.5.0`, `databricks-sdk==0.170.0`,
  and `pytest==9.2.0`, with `poetry.lock` regenerated so every transitive
  package is at its latest release. The dependency-freshness record now holds
  `vllm` and its hash-locked 0.30.0 closure with Databricks-validation reasons
  (vLLM 0.34.0 publishes no CUDA 12.9 wheel).
- PR evidence records which model reviewed the pull request. The record type
  is now `document_kv.pr_evidence.v2`: the `gpt55_review_*` fields and
  `--gpt55-review-*` flags become `review_*` and `--review-*`, a non-empty
  `review_model` (`--review-model`) is required, and `GPT55_REVIEW_OUTCOMES`
  is renamed `PR_REVIEW_OUTCOMES`.
- vLLM release benchmark: `cachet-vllm-smoke --release-raw-dataset` (also
  through `cachet-vllm-smoke-databricks-job`) runs a V1 plan that generates
  Vanilla handoffs, runs each arm against its own vLLM server with cold,
  attested payload reads, merges the arms with the new
  `benchmark_runner merge-isolated` subcommand, gates the comparison on latency
  under the publication policy, and writes the plan-execution record the strict
  release bundle requires. Plans gain `--isolated-arm-jobs`,
  `--benchmark-handoff-cache-method`, `--benchmark-handoff-segment-per-document`,
  and `--benchmark-handoff-generator-version`; generated bundles write their
  artifact identity (`--artifact-identity-output-json`); cache-state
  attestations can be read from the vLLM provider's telemetry `.jsonl` log; and
  `execute_benchmark_job_plan` takes `command_context` and `env`.
- vLLM 0.30.0 (CUDA 12.9) serving profile, with the pinned hash-locked vLLM
  runtime lock shipped in the wheel, and `vllm_wheel_repack` for building the
  E5M2-patched vLLM wheel.
- Registered method framework: method lifecycles, artifact, token, and layout
  contracts, N-way evaluation arms, paired statistics, benchmark gates and
  metrics, Quality v2 scoring, and a method scaffold and
  conformance checker (`cachet-method-scaffold`, `cachet-method-conformance`).
- Vanilla v2: per-document pre-RoPE keys with absolute-position injection, and
  a single-pass segmented vLLM cold loader.
- Decode throughput is now aggregated as total completion tokens over total
  post-TTFT decode time; the method `vanilla` is renamed `vanilla_prefill`, and
  connector actions take an explicit reuse plan.
- The packaged vLLM Databricks smoke runner installs the official vLLM
  `0.30.0+cu129` wheel from the CUDA 12.9 indexes, and both the vLLM and SGLang
  smoke runners emit handoffs that carry the registered artifact and token
  contracts. The vLLM runner passes the registered cache arm that those
  handoffs declare. On 2026-11-18 and 2026-11-19
  the vLLM and SGLang runners each passed on L4 and A10G.
- `packaging` pinned to 26.3.
- Benchmark records: the vLLM 0.30.0 campaign report and gate, the Quality v2
  report, and the plan-coherent auxiliary latency re-run, plus
  `benchmarks/verify_sealed_records.py`, a dependency-free integrity check of
  those records and the README regions rendered from them.
- The campaign machinery that produced the records (resource ledger, GPU
  qualification, latency, full-score, and finalizer stages, source freeze,
  LMCache/hybrid launchers, and the full offline validators) is not part of
  the package; it is preserved at the `vllm-0300-campaign-archive` tag.
- Removed the superseded `benchmarks/current/` summary and the earlier result
  folders and indexes under
  `benchmarks/{databricks,native-engine,sglang,storage,vllm}/`.
- Sanitized benchmark evidence keeps the final-answer parser identity, status,
  and NIAH cell id, and records a SHA-256 of the extracted answer instead of
  the answer, so scored results can pass the publication gate after
  sanitization.
- Scorer manifests can declare a non-v1 final-answer parser (the Quality v2
  scorers do), and the publication gate checks measurements against the
  declared parser. Manifests for v1 scorers are unchanged.
- An integral `rope_theta` now yields the same artifact id as the equal float,
  and reuse-plan and runtime-operation digests reject NaN and Infinity.
- Removed legacy and dead public API:
  - the `legacy_compatibility`, `scheduler`, and `runtime_telemetry` modules,
    `vllm_kv_injection.transfer_config_cli`, and
    `sglang_kv_injection.hicache_config_cli`; `AdmissionQueue` and
    `PreparedRequest` remain in `admission` and the package root;
  - the `legacy_migration_evidence` release-bundle role and
    `--legacy-migration-evidence-json`;
  - the `resource` measurement scope and `BenchmarkResourceEvidence`;
    experiment manifests no longer carry resource fields, so manifest digests
    change;
  - `CacheGenerationMethod.LMCACHE`, the built-in `lmcache` method, and the
    MultiConnector transfer-config helpers;
  - the 31 `document-kv-*` console scripts (use the `cachet-*` names);
    Databricks run names, purpose tags, and workspace folders now use
    `cachet-*`, so run-status files recorded under the old purposes fail
    strict coverage;
  - the Databricks `g5` alias names and the `aws_g5_node_type` run-status
    field;
  - the vLLM 0.27.1 repack path (`repack_vllm_0271_cu129_wheel` is now
    `repack_vllm_cu129_wheel`), the five-argument engine-package `KVSegment`
    constructor, SGLang HiCache `exist`, `selected_documents`, unwritten fp8
    dtype aliases,
    `five_deployment_paired_statistics`, and unused helpers and constants.
- Readers accept only current formats: engine handoff schema 4, benchmark
  handoff manifest schema 3, manifests with per-arm runtime environments,
  native-probe and release-evidence sidecars with every current field, and
  engine-probe target files with the full envelope. Gate configs drop
  `max_exact_match_drop` and `max_answer_found_drop`, and a quality claim
  without a manifest fails.
- The debug path for unregistered methods and uncontracted artifacts is named
  for what it does: `--allow-uncontracted-artifact`,
  `--benchmark-handoff-allow-uncontracted-artifact`, and
  `allow_unregistered_method=`. Segmented-load telemetry reports
  `segment_remerge_reason` and the `segment_remerge` strategy, and vLLM
  cache-state attestations drop the duplicate `expected_stored_bytes` key
  (read `expected_bytes`).
- The release-bundle wheel check accepts only the four Cachet packages and
  exactly the console scripts in `pyproject.toml`.
- `pyspark` is no longer in the `databricks` extra.
- Removed committed historical evidence: `docs/release-ops/pr-evidence/` (the
  release PR's evidence sidecar is generated when the bundle is built),
  `docs/release-ops/benchmark-archive/`, and the legacy-migration evidence.
- SGLang probe handles keep each segment's token contract.
- Release evidence accepts the benchmark plan's
  `--benchmark-evidence-gate-json`, so planned release-evidence steps no longer
  fail on an unknown flag. The gate must pass under the publication policy and
  match both the benchmark record's embedded gate and a gate recomputed from
  the record's `gate_inputs`. Strict release bundles require and re-verify it,
  and benchmark plans that request release evidence must use
  `--evidence-policy publication`.
- The benchmark page no longer publishes the campaign's v1 full-score tables
  (dataset scores and the NIAH grid), whose strict parser rejected most outputs
  of both methods. Quality Benchmark v2 is the published answer-quality
  measure. The sealed campaign report is unchanged and still contains the v1
  pass; `verify_sealed_records.py` now checks 11 README regions.
- Quality v2 is the only answer-quality protocol. The runner, handoff
  generation, benchmark plans, live checks, both smoke runners, and their
  Databricks jobs default to its Qwen3 chat prompt (template
  `v3-qwen3-chat-final-answer`), a 256-token decode at temperature 0, the
  `cachet.first_final_answer` parser, and `quality_v2_scorer_registry()`. The
  v1 prompt, scorers, registry, and synthetic NIAH generator are removed, and
  strict whole-output `<final_answer>` compliance is only a format metric.
  - Publication-approved scorers must declare a prompt and an answer parser.
    The publication gate re-runs the declared parser, and release evidence
    re-scores every measurement with its dataset's Quality v2 scorer.
  - `cachet.benchmark_dataset_sources` stages NIAH with
    `build_quality_v2_niah_records` over the HotpotQA and MuSiQue source
    paragraphs, reproducing the sealed Quality v2 NIAH input. `dataset_prep`
    no longer generates NIAH, and its haystack and needle flags are removed.
  - Live checks judge the parsed `<final_answer>` block rather than a substring
    of the raw output. The live-check prompt-format option
    (`--prompt-format`, `--live-check-prompt-format`) is removed: completion
    requests send the chat-framed prompt and chat requests send its user turn.
  - The new default prompt template version changes artifact ids and cache
    keys, so handoffs generated under the v1 template must be regenerated.

## 0.2.0

- Cachet-branded `cachet-kv` distribution and `cachet` import facade.
- Document-oriented request, manifest, storage, materialization, and workflow
  APIs.
- Local memory and disk storage readers plus Unity Catalog Volume support.
- vLLM and SGLang adapter modules shipped in the same distribution.
- Human-readable benchmark summaries in `benchmarks/README.md`.
- Public quickstart and beginner documentation.
