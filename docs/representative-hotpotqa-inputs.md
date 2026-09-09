# Representative HotpotQA inputs

The representative vLLM canaries and cold-loader ablations use a canonical
`hotpotqa.jsonl` whose logical Cachet prefill prompts contain exactly 8,192,
16,384, or 32,768 tokens. Prepare that file before generating full-prefix and
vanilla handoffs:

```bash
cachet-prepare-representative-hotpotqa \
  --source /data/hotpot_dev_distractor_v1.json \
  --output-jsonl /data/8k/hotpotqa.jsonl \
  --provenance-json /data/8k/hotpotqa.provenance.json \
  --input-tokens-target 8192 \
  --example-count 2
```

`--source` accepts the official HotpotQA dev-distractor JSON array or an
already-canonical Cachet HotpotQA JSONL. Selection is stable by normalized
example identity and canonical source-record digest. Each selected example has
at least two documents; when it is short, the command adds one deterministic,
irrelevant padding document. It then reloads the output and fails unless every
`build_prompt_parts(example).prefill_prompt` is exactly the requested size.

The command hard-pins `Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554` and always tokenizes with
`add_special_tokens=False`. It imports `AutoTokenizer` lazily and never loads a
causal model. Run it in an environment that contains Transformers; the serving
environment used by the representative canary already does.

The provenance JSON contains tokenizer/prompt-contract pins, counts, and
SHA-256 digests. It deliberately excludes source/output paths, example IDs,
questions, answers, documents, padding text, and raw prompts. Pass the emitted
JSONL and the same pinned tokenizer settings to
`prepare_representative_canary_inputs`; retain the provenance beside the staged
dataset so the exact source and output bytes remain auditable.
