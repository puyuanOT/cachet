# Vanilla score canary protocol

This appendix defines a small matched correctness canary for Baseline and
Vanilla KV. It is intentionally **not** a full-dataset benchmark and is not
publication evidence. With only five selected examples per dataset, results
are descriptive diagnostics; they must not be generalized beyond these exact
content-addressed inputs.

The machine-readable source of truth is
`document_kv_cache.score_canary`. Preparation writes a manifest that binds the
complete protocol, raw-source identities, selected records, exact prompts, and
runner-compatible arm settings.

## Frozen logical inputs

| Field | Value |
| --- | --- |
| Protocol | `vanilla-score-canary-8k-n5-v1` |
| Datasets | Biography, HotpotQA, MusiQue, NIAH |
| Examples | 5 per dataset, 20 per arm |
| Selection | Ascending SHA-256 of `seed\0dataset\0canonical-source-record-SHA-256` |
| Context | Exactly 8,192 tokens including system prompt, documents, and question |
| Tokenizer | `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Tokenization | `add_special_tokens=false`; prompt template `v1-benchmark`; system prompt at start |
| Length control | Preserve every real document and append one deterministic irrelevant padding document; never truncate a real document |
| Decode | `max_tokens=64`, temperature 0, streaming, natural EOS; `ignore_eos` is not sent |
| Repeats / parallelism | 1 / 4 |

Selection is independent of source row order. A record is eligible only when
it has an expected answer, its unpadded prompt is no longer than the target,
deterministic padding can reach the target exactly, independently tokenizing
every `benchmark_cache_prefix_segments(example)` segment composes to the exact
cache-prefix token IDs, and that cache prefix is a strict token prefix of the
full prompt. The manifest records per-segment text/token hashes and counts,
the composed/cache-prefix/full-prompt token hashes and counts,
the raw source byte hash, canonical selected-record hashes, unpadded and
prepared prompt hashes, exact token counts, padding attestations, output byte
hashes, protocol hash, and logical suite hash. Prepared filenames contain the
full output SHA-256.

## Metrics and claim boundary

| Dataset | Primary metric | Status |
| --- | --- | --- |
| Biography | `answer_found` | Cachet diagnostic |
| HotpotQA | answer F1 | Pinned official answer scorer |
| MusiQue | `answer_found` | Cachet diagnostic |
| NIAH | exact match | Cachet diagnostic |

Report only each arm's mean primary metric over the exact five examples. Keep
per-example outputs for the matched audit. Do not report confidence intervals,
significance, superiority, full-dataset accuracy, or publication claims from
this canary. Only HotpotQA's scorer is publication-approved; the sample size is
still far too small for a publication claim.

## Current matched diagnostic

The two isolated jobs completed all 20 requests per arm at request parallelism
4. The results are descriptive/nonpublication evidence and are not substituted
into the full-dataset score table.

| Method | Biography answer-found | HotpotQA answer F1 | MusiQue answer-found | NIAH exact match |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1.000000 | 0.108975 | 0.200000 | 0.000000 |
| Vanilla&nbsp;KV | 1.000000 | 0.040827 | 0.000000 | 0.000000 |

The [sanitized evidence record](../main-vanilla-descriptive-evidence/evidence.json)
contains the exact unrounded means, matched suite and manifest identities,
source revision, frozen scorer wheel identity, and closed validation proof.

## Isolated jobs

Run exactly two fresh single-node `g6.8xlarge` jobs. Both use the same 20
prepared records, model/tokenizer revision, 4-bit bitsandbytes weights, vLLM
0.23.0, Q8 (`fp8_e5m2`) runtime KV, parallelism four, decode settings, ordering,
and per-request prefix-cache salt. Do not combine the arms in one server.

1. Baseline: `baseline_prefill`, with no handoff generation.
2. Vanilla: `vanilla_prefill`, with
   `build_pre_rope_transformers_kv_chunk_generator`, one handoff segment per
   document, `fp8_e5m2` payloads, local-NVMe storage, payload cache disabled,
   and `DOCUMENT_KV_EVICT_PAGE_CACHE=1`.

Vanilla stores independently computed pre-RoPE keys and values, assembles the
documents in logical order, and applies each token's true absolute position at
injection. The Vanilla job enriches a private copy of the JSONL at run time;
the immutable logical JSONL is shared with Baseline and contains no
`kv_transfer_params` or arm-specific transfer metadata.

At Qwen3-4B's 73,728 Q8 KV bytes per token, the nominal Vanilla artifact upper
bound is 12,079,595,520 bytes (11.25 GiB), which fits local NVMe. A prior
calibration took about 735 seconds to generate two 8k pre-RoPE examples; linear
planning projects 7,350 seconds for all 20. The run plan therefore allows four
hours for Vanilla and two hours for Baseline: at most six reserved
single-node cluster-hours, with roughly three to four expected.

## Prepare and validate

Starting from the repository root:

```bash
poetry run python -m document_kv_cache.score_canary prepare \
  --source biography=/path/to/biography.jsonl \
  --source hotpotqa=/path/to/hotpotqa.jsonl \
  --source musique=/path/to/musique.jsonl \
  --source niah=/path/to/niah.jsonl \
  --output-dir /path/to/content-addressed-score-canary
```

Revalidate before upload and again from the staged copy:

```bash
poetry run python -m document_kv_cache.score_canary validate \
  --manifest /path/to/content-addressed-score-canary/vanilla-score-canary-8k-n5-v1-manifest.json \
  --source biography=/path/to/biography.jsonl \
  --source hotpotqa=/path/to/hotpotqa.jsonl \
  --source musique=/path/to/musique.jsonl \
  --source niah=/path/to/niah.jsonl
```

The manifest contains the common `document_kv_cache.vllm_smoke` arguments,
the validated Vanilla arm JSON, both isolated job identities, handoff settings,
storage estimate, and time budget. Replace its path placeholders with the
manifest's content-addressed output filenames; do not change any other frozen
field without creating a new protocol ID.
