# Benchmarks

Maintainer reference for the V1 benchmark contract: handoff generation,
command plans, dataset preparation, the OpenAI-compatible runner, and the
storage-reader benchmark. Published results live in
[`benchmarks/README.md`](../../../benchmarks/README.md); what may be committed as
evidence is defined in [`../../evidence-policy.md`](../../evidence-policy.md).
Running these commands as Databricks jobs is covered in
[`databricks.md`](databricks.md), and validating their outputs in
[`release-and-governance.md`](release-and-governance.md).

## Benchmark Contract

- The V1 surface targets Biography, HotpotQA, MusiQue, and NIAH with Qwen3 4B
  Instruct, and compares document KV-cache reuse against the no-cache
  `baseline_prefill` arm.
- `aws-g6-l4` (plain AWS `g6.8xlarge`) is the default and release target.
  `aws-g5-a10g` (`g5.8xlarge`) is an explicit non-default compatibility target.
- The published tables in `benchmarks/README.md` are rendered from sealed
  appendix records; `python benchmarks/verify_sealed_records.py` checks them.
  Do not hand-edit sealed records or the governed README regions, and add a new
  appendix folder only after sanitized evidence passes the publication gate.
- Provider-backed SGLang probes are integration checks. They do not populate a
  latency or score table, and the Q8 pre-RoPE SGLang serving path stays
  explicitly unsupported.
- The package does not own model execution. Databricks jobs, vLLM or SGLang
  adapters, or local harnesses provide `BenchmarkEngine` implementations;
  Cachet keeps dataset parsing, prompt pairing, error capture, summaries, and
  baseline comparison consistent. Per-module responsibilities are listed in
  [`src/document_kv_cache/README.md`](../../../src/document_kv_cache/README.md).
- `evaluate_v1_benchmark_evidence` marks a report release-evaluable only when
  all four datasets, only the expected baseline and cache arms, comparisons
  with speedups and quality deltas, successful latency measurements, and
  quality rates are present. Single-dataset smoke runs are allowed but report
  `v1_evidence.ok` as false.
- Every benchmark path uses the Quality v2 answer-quality protocol
  (`cachet.quality_v2`): one Qwen3 chat user turn per request (prompt template
  `v3-qwen3-chat-final-answer`), decoding at temperature 0 for up to 256 tokens,
  the tolerant `<final_answer>` parser, and the scorer that
  `quality_v2_scorer_registry()` registers for each dataset. The runner,
  handoff generation, command plans, smoke runners, and Databricks job helpers
  default to it. Release evidence re-scores every measurement's output with
  its dataset's Quality v2 scorer and rejects measurements that declare another
  scorer or carry scores or labels that re-scoring does not reproduce.
  Strict whole-output `<final_answer>` compliance is only a format metric.
- Quality v2 is a library API with no packaged CLI: besides the prompts,
  parser, and scorers it supplies the NIAH builder and the report and
  README-region validators. The job runner that executed the sealed campaign is
  preserved at the `vllm-0300-campaign-archive` tag.

For servers exposing the OpenAI-compatible completions API, use the
stdlib-only `OpenAICompatibleCompletionEngine`. It posts the full logical
prompt by default, which is correct for ordinary vLLM/SGLang servers, native
vLLM connector scheduling, and platform-managed prefix caching. Set
`prompt_text_mode="runtime"` only when the endpoint is a KV-aware adapter or
proxy that binds the cached prefix out of band and expects only the runtime
suffix. Streaming responses provide TTFT; non-streaming responses report TTFT
equal to total completion latency.

## Handoff Bundles

`cachet-benchmark-handoff-bundles` generates per-example handoff JSON and
payload files from prepared rows through a caller-supplied
`KVChunkGenerator` factory whose payload geometry matches the model layout.
Cachet does not ship a fake benchmark generator.

- For Vanilla generation in a Transformers environment, use
  `document_kv_cache.transformers_generator:build_pre_rope_transformers_kv_chunk_generator`
  with `--cache-method vanilla_prefill --segment-per-document`. For the
  monolithic full-prefix control, use
  `build_post_rope_transformers_kv_chunk_generator` and omit those two options.
- Use a floating KV dtype such as `--dtype bfloat16` for model-produced
  payloads.
- Set `CACHET_TRANSFORMERS_TRUST_REMOTE_CODE=true` only for model repositories
  that require custom Transformers code. The generator expects Hugging Face
  head-major cache tensors; token-major runtimes can set
  `CACHET_TRANSFORMERS_CACHE_AXIS_ORDER=token_major`.
- The CLI derives the default `qwen3:4b-instruct` layout from the built-in
  profile. Custom models pass `--layout-version`, `--dtype`, `--num-layers`,
  `--block-size`, and `--bytes-per-token`, plus `--num-query-heads`,
  `--num-kv-heads`, `--head-size`, `--kv-stride-bytes`, `--shares-kv-storage`,
  and `--storage-layout` when the adapter needs attention geometry.
- `--artifact-identity-output-json` writes the generated artifact's
  `ArtifactIdentity`, the input publication gating takes as
  `--artifact-identity-json`.

```bash
CACHET_TRANSFORMERS_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 \
CACHET_TRANSFORMERS_DEVICE=cuda \
cachet-benchmark-handoff-bundles \
  --input-jsonl /data/v1-prepared/biography.jsonl \
  --output-dir /Volumes/catalog/schema/volume/cachet/handoffs \
  --output-manifest-json /data/handoffs/biography-manifest.json \
  --generator-factory document_kv_cache.transformers_generator:build_pre_rope_transformers_kv_chunk_generator \
  --cache-method vanilla_prefill \
  --segment-per-document \
  --dtype bfloat16
```

For SGLang prepared runs, add `--backend sglang --sglang-hicache-page-size 16`
to write `document_kv.sglang_hicache_page_keys` into the manifest in the same
pass. The generator must expose the tokenizer used for the cached prefix. The
page size must match the KV layout block size, keys are emitted only for full
payload pages, and they match the Quality v2 cache prefix sent as completion
text; run the prepared SGLang path with `--live-check-request-mode completion`,
or supply page keys for the chat-template-rendered prefix through the manifest
template when it uses chat request mode.

When handoff JSON files already exist, build the closed manifest the benchmark
plan consumes. The template supports `{dataset}` and `{example_id}`, and every
referenced handoff is validated before the manifest is written. For SGLang, add
`--sglang-hicache-page-keys-json-template` pointing at per-example JSON arrays
(or objects containing `document_kv.sglang_hicache_page_keys`) holding the
exact HiCache page-hash chain for the cached prefix.

```bash
cachet-benchmark-handoff-manifest \
  --input-jsonl /data/v1-prepared/biography.jsonl \
  --handoff-json-template '/Volumes/catalog/schema/volume/cachet/handoffs/{dataset}/{example_id}.handoff.json' \
  --expected-backend vllm \
  --output-json /data/handoffs/biography-manifest.json
```

## Benchmark Plan

To run the V1 benchmark contract against existing OpenAI-compatible vLLM or
SGLang servers, generate a reproducible command plan for all four datasets. The
plan-level generator below is the monolithic full-prefix control; to generate
Vanilla handoffs in the plan, use the pre-RoPE factory with
`--benchmark-handoff-cache-method vanilla_prefill --benchmark-handoff-segment-per-document`:

```bash
python -m document_kv_cache.benchmark_plan \
  --raw-dataset biography=/raw/biography.jsonl \
  --raw-dataset hotpotqa=/raw/hotpotqa.jsonl \
  --raw-dataset musique=/raw/musique.jsonl \
  --raw-dataset niah=/raw/niah.jsonl \
  --prepared-dir /data/v1-prepared \
  --benchmark-handoff-manifest-json biography=/data/handoffs/biography-manifest.json \
  --benchmark-handoff-manifest-json hotpotqa=/data/handoffs/hotpotqa-manifest.json \
  --benchmark-handoff-manifest-json musique=/data/handoffs/musique-manifest.json \
  --benchmark-handoff-manifest-json niah=/data/handoffs/niah-manifest.json \
  --benchmark-handoff-generator-factory document_kv_cache.transformers_generator:build_post_rope_transformers_kv_chunk_generator \
  --benchmark-handoff-dtype bfloat16 \
  --benchmark-handoff-output-dir /data/handoffs/generated \
  --base-url http://localhost:8000 \
  --storage-benchmark-workspace-dir /local_disk0/cachet-storage-benchmark \
  --storage-benchmark-uc-volume-root /Volumes/catalog/schema/volume/cachet-storage-benchmark \
  --engine-probe-handoff-json vllm=/data/vllm-handoff.json \
  --engine-probe-output-json vllm=/data/vllm-engine-probe.json \
  --engine-probe-actions-output-json vllm=/data/vllm-connector-actions.json \
  --engine-probe-native-probe-factories-output-json vllm=/data/vllm-native-probe-factories.json \
  --engine-probe-handoff-json sglang=/data/sglang-handoff.json \
  --engine-probe-output-json sglang=/data/sglang-engine-probe.json \
  --engine-probe-actions-output-json sglang=/data/sglang-connector-actions.json \
  --engine-probe-native-probe-factories-output-json sglang=/data/sglang-native-probe-factories.json \
  --engine-probe-sglang-runtime-preflight-output-json sglang=/data/sglang-runtime-preflight.json \
  --engine-probe-sglang-runtime-preflight-launch-config-json sglang=/data/engine-launch-configs/sglang-launch-config.json \
  --engine-probe-vllm-runtime-preflight-output-json vllm=/data/vllm-runtime-preflight.json \
  --engine-probe-vllm-runtime-preflight-layer-names-json vllm=/data/vllm-layer-names.json \
  --engine-probe-native-delegate-factory vllm=vllm_kv_injection.probe:build_native_connector_probe \
  --engine-probe-native-delegate-factory sglang=sglang_kv_injection.probe:build_native_connector_probe \
  --engine-probe-metadata vllm=vllm_kv_injection.connector_factory=vllm_kv_injection.probe:build_document_kv_native_probe_connector \
  --engine-probe-metadata sglang=sglang_kv_injection.connector_factory=sglang_kv_injection.probe:build_document_kv_hicache_probe_connector \
  --engine-probe-use-builtin-factories \
  --evidence-policy publication \
  --release-evidence-output-json /data/release-evidence.json \
  --release-preflight-output-json /data/release-inputs.json \
  --release-bundle-output-dir /data/cachet-release-bundle \
  --release-bundle-output-json /data/release-bundle-manifest.json \
  --release-bundle-databricks-run-status-json /data/databricks-run-status-benchmark.json \
  --release-bundle-databricks-run-status-json /data/databricks-run-status-storage.json \
  --release-bundle-databricks-run-status-json /data/databricks-run-status-vllm-engine-probe.json \
  --release-bundle-databricks-run-status-json /data/databricks-run-status-sglang-engine-probe.json \
  --release-bundle-package-wheel /data/dist/cachet_kv-0.3.0-py3-none-any.whl \
  --release-bundle-compatibility-benchmark-json /data/g5-v1-results.json \
  --release-bundle-pr-evidence-json /data/pr-evidence.json \
  --github-governance-output-json /data/github-governance.json \
  --repository-hygiene-output-json /data/repository-hygiene.json \
  --native-probe-factories-output-json /data/native-probe-factories.json \
  --engine-launch-config-output-dir /data/engine-launch-configs \
  --engine-probe-targets-output-json /data/engine-probe-targets.json \
  --engine-probe-targets-release-safe \
  --plan-output-json /data/v1-plan.json \
  --plan-output-sh /data/run-v1-benchmark.sh
```

- The generated script runs `dataset_prep` for each raw file, generates
  handoff bundles when `--benchmark-handoff-generator-factory` is set, enriches
  datasets with handoffs through `document_kv_cache.benchmark_handoffs`, and
  then runs `benchmark_runner`. Enriched inputs default to
  `<prepared-dir>/<dataset>.handoffs.jsonl` unless
  `--benchmark-handoff-output-jsonl DATASET=PATH` is set. Generated bundles
  also write `<handoff-output-dir>/<dataset>-artifact-identity.json`, which the
  runner receives as `--artifact-identity-json`; resolved
  `--model-revision`, `--tokenizer-id`, `--tokenizer-revision`, and
  `--benchmark-handoff-generator-version` values are stamped into those
  identities.
- `--isolated-arm-jobs` (requires `--runtime-id`) replaces the single runner
  command with one `run-benchmark-<arm>` command per arm, each writing a
  one-arm record to `<benchmark-output>.<arm>.json` under runtime id
  `<runtime-id>:<arm>`, then a `merge-benchmark-arms` command that runs
  `benchmark_runner merge-isolated` to produce the combined record and
  evidence gate. This is the `separate_process_or_job` isolation the
  publication policy accepts. Like `benchmark_runner --output-json`, the merged
  record keeps raw answers and outputs because release evidence re-scores them;
  `merge-isolated --sanitize-evidence` writes the sanitized form instead. Each arm needs its own serving process: the
  executor's `command_context` hook (used by the vLLM release benchmark below)
  starts and stops one per arm command, and any other caller must do the same.
- `--cache-state-attestation-json` accepts attestation JSON, vLLM telemetry
  JSON, or the vLLM provider's telemetry `.jsonl` log; only
  `document_kv.vllm_native_provider_load.v1` lines are read from a log. Cold
  latency claims need one attestation per cache request.
- `--storage-benchmark-workspace-dir` appends a storage-reader benchmark on the
  same node; the Unity Catalog reader is added only when
  `--storage-benchmark-uc-volume-root` points at a real UC Volume. Output goes
  to `<prepared-dir>/<suite-id>-storage-benchmark.json` unless
  `--storage-benchmark-output-json` is set.
- Backend-keyed `--engine-probe-*` options append native
  `document_kv_cache.engine_probe` commands, and release evidence consumes
  their outputs. Planned probes consumed by release evidence must include
  `--engine-probe-actions-output-json` for each backend and cannot use the
  debug-only `--engine-probe-engine-version` or
  `--allow-non-native-engine-probe`. If native probe and connector-action
  records already exist, skip the planned probe flags and pass them directly
  with repeatable `--release-engine-probe-json` and
  `--release-engine-actions-json`.
- `--engine-probe-use-builtin-factories` fills missing factories with the
  package-owned vLLM/SGLang paths; pass
  `--engine-probe-factory BACKEND=MODULE:CALLABLE` for a downstream adapter.
- The native delegate factories and connector metadata make each target
  provider-backed, so the Databricks probe job installs that backend's pinned
  runtime (the patched vLLM wheel, `sglang==0.5.10.post1`). Without them a
  release-safe target file still validates, but its probe tasks install no
  engine to probe.
- `--engine-probe-fixture-output-dir BACKEND=DIR` first writes a deterministic
  Qwen3 V1 probe fixture; when `--engine-probe-handoff-json` is omitted for that
  backend, the handoff is `DIR/qwen3-v1-fixture.handoff.json` and the actions
  sidecar defaults to `DIR/qwen3-v1-fixture.actions.json`. Select the payload
  format with `--engine-probe-fixture-payload-mode BACKEND=merged|segmented`;
  built-in factories default to `merged`.
- `--github-governance-output-json`, `--repository-hygiene-output-json`,
  `--native-probe-factories-output-json`, `--release-preflight-output-json`,
  and `--engine-launch-config-output-dir` generate those sidecars inside the
  plan, and a planned release bundle includes them automatically. Use the
  matching `--release-bundle-*` input flag only for sidecars produced outside
  the plan. Explicit `--release-bundle-engine-launch-config-json` paths must
  match the generated launch configs when both are given.
- `--engine-probe-targets-output-json` writes a
  `document_kv.engine_probe_targets.v1` file for the Databricks engine-probe
  job. `--engine-probe-targets-release-safe` requires exactly one native vLLM
  probe and one native SGLang probe, rejects debug-only probe settings, and
  carries each backend's `actions_output_json` path so the probe jobs write the
  sidecars release evidence needs. The remaining per-target requirements are in
  [`databricks.md`](databricks.md#engine-probe-jobs).
- With release-evidence flags, the plan appends
  `document_kv_cache.release_evidence` last; `--release-bundle-output-dir`
  appends `document_kv_cache.release_bundle` after it. Both receive the
  benchmark's evidence gate as `--benchmark-evidence-gate-json`, written to
  `--evidence-gate-output-json` (default
  `<prepared-dir>/<suite-id>-evidence-gate.json`).
- The plan executor treats plan JSON as a closed schema and rejects unknown
  top-level or per-command keys before running anything. It writes
  `plan-execution.json` only after the planned commands finish, and the
  planner has no input for the dependency freshness sidecar, so build the
  final strict bundle as a separate `cachet-release-bundle --require-complete-v1`
  command (see [Release Bundle](release-and-governance.md#release-bundle)).

### vLLM Release Benchmark

`cachet-vllm-smoke --release-raw-dataset DATASET=PATH` (repeat for all four V1
datasets) runs the release benchmark as a plan on one GPU node:

1. It writes `release-benchmark-plan.json`, which prepares the raw rows,
   generates Vanilla handoffs (`vanilla_prefill`, one segment per document,
   bfloat16, Qwen3 revision `cdbee75f17c01a7cc42f958dc650907174af0554`) under
   `<local-root>/<benchmark-id>/handoffs`, and attaches them.
2. Each arm runs against a fresh vLLM server: `baseline_prefill` with a unique
   prefix-cache salt per request, and the cache arm with the payload cache
   disabled and `DOCUMENT_KV_EVICT_PAGE_CACHE=1`, so every load is a cold read
   attested in the provider telemetry log.
3. The merged record is gated under the publication policy for the `latency`
   measurement scope, with every cache request cold-attested.
4. It writes `release-plan-execution.json` with the plan's `plan_source`, the
   plan-execution record the strict release bundle requires.

The run fails if any planned command fails or the merged evidence gate is not
ok. The release gate claims latency only. Answer quality is reported from the
sealed [Quality v2 report](../../../benchmarks/appendix/vllm-0300-quality-v2/README.md),
where Vanilla does not meet answer-quality non-inferiority on HotpotQA or
MusiQue. `--max-model-len` defaults to 40960 so the 32K-token NIAH cell fits.
The vLLM provider copies each cache request's payload to the GPU in one piece,
outside vLLM's memory budget (about 4.5 GiB for a 32K-token NIAH document), so
release runs default to `--gpu-memory-utilization 0.70`, which 24 GB GPUs (L4,
A10G) need; at 0.80 that copy runs out of GPU memory. Each dataset's prompts are
checked against the context budget before its handoffs are generated. Handoffs
are stored twice (build shard and per-request payload), at about 147 KB per
token for Qwen3 4B in bfloat16. The default `--release-limit-per-dataset 54`
takes the first 54 rows of each staged dataset (six per NIAH cell), about 1.1
million document tokens, which need about 330 GB on `/local_disk0`; full
datasets do not fit. `--release-limit-per-dataset`, `--release-warmups`,
`--release-model-revision`, and `--release-generator-factory` adjust the plan.
The Databricks job runs only the `vanilla_prefill` method.

## Dataset Prep And Runner

NIAH has no raw source file. Stage it, together with the other three datasets,
from the public HotpotQA, MusiQue, and WikiBio sources:

```bash
python -m cachet.benchmark_dataset_sources --output-dir /data/raw
```

The NIAH grid comes from `build_quality_v2_niah_records` over every HotpotQA,
then MusiQue, source paragraph in file order, with prompt tokens counted by the
Qwen3 tokenizer at revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
Staging needs `pyarrow` and `transformers`, plus `huggingface_hub` unless
`--hotpotqa-parquet`, `--musique-jsonl`, and `--wikibio-zip` point at local
copies. `dataset-source-metadata.json` records the builder, seed, and token
counter. The staged `biography.jsonl`, `hotpotqa.jsonl`, `musique.jsonl`, and
`niah.jsonl` files are valid `--input-jsonl` and plan `--raw-dataset` inputs.

Staging the default sources (`hotpotqa/hotpot_qa` distractor validation and
`voidful/MuSiQue` `musique_ans_v1.0_dev.jsonl`) with the pinned tokenizer
reproduces the sealed Quality v2 NIAH input: `niah.jsonl` has SHA-256
`d6c2c9d988e1417c1434cc565602a4b2c4334f87808e96e7eed98caad69f2c39`, as listed
in the [Quality v2 appendix](../../../benchmarks/appendix/vllm-0300-quality-v2/README.md)
(verified with `transformers` 5.12.1).

Convert a single raw dataset without a plan:

```bash
python -m cachet.dataset_prep \
  --dataset hotpotqa \
  --input-jsonl /data/raw_hotpotqa.jsonl \
  --output-jsonl /data/hotpotqa.jsonl
```

Malformed rows fail with the physical JSONL line number. Then run the
benchmark directly or execute the generated plan script:

```bash
python -m document_kv_cache.benchmark_runner \
  --dataset biography=/data/biography.jsonl \
  --dataset hotpotqa=/data/hotpotqa.jsonl \
  --dataset musique=/data/musique.jsonl \
  --dataset niah=/data/niah.jsonl \
  --base-url http://localhost:8000 \
  --model-id qwen3:4b-instruct \
  --hardware-target aws-g6-l4 \
  --output-json v1-results.json
```

- With the default `/v1/completions` endpoint, `--base-url` may be a server
  root or an OpenAI-style `/v1` base. Custom `--endpoint` and
  `--cache-endpoint` values are appended to the exact base URL.
- Both arms post the full logical prompt by default. Add `--cache-base-url` and
  `--cache-runtime-prompt` only when that cache endpoint is a KV-aware adapter
  or proxy that binds the cached prefix out of band.
- `--max-tokens` defaults to the Quality v2 budget of 256 and `--temperature`
  to 0.

## Storage-Reader Benchmark

The synthetic reader benchmark writes one packed shard and reports p50/p95
read latency and throughput for Memory, Disk, and Unity Catalog Volume readers:

```bash
python -m document_kv_cache.storage_benchmark \
  --workspace-dir /local_disk0/cachet-storage-benchmark \
  --uc-volume-root /Volumes/catalog/schema/volume/cachet-storage-benchmark \
  --chunk-count 64 \
  --chunk-bytes 1048576 \
  --repeats 4 \
  --parallelism 8 \
  --output-json storage-benchmark.json
```

Omit `--uc-volume-root` for a local UC-reader smoke test, or repeat `--reader`
to select backends such as `--reader memory --reader disk`. The output has
`storage_evidence` for the selected readers and `release_storage_evidence` for
the strict Memory + Disk + real Unity Catalog Volume release gate; both report
missing readers, reader errors, absent metrics, and whether the UC root is a
`/Volumes/<catalog>/<schema>/<volume>` path when required.
