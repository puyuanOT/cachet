# Getting Started

This guide is for new Cachet users who want to understand the local workflow
before touching serving engines, cloud benchmarks, or release-audit material.

## Install

```bash
pip install cachet-kv
```

For a source checkout:

```bash
git clone https://github.com/puyuanOT/cachet.git
cd cachet
python examples/quickstart_local.py
```

The source-checkout example adds `src/` to `sys.path` for convenience, so you do
not need an editable install just to try it.

## Run The Local Quickstart

After `pip install cachet-kv`:

```bash
python -m cachet.quickstart_local
```

From a source checkout:

```bash
python examples/quickstart_local.py
```

Both commands use the same tiny fake KV generator and toy layout. They write
one local disk `.kvpack` shard, write one memory-backed shard, materialize a
request, and build an engine-ready handoff record.

You should see output like:

```text
generated chunks: 4
materialized bytes: 340
materialized tiers: cold_storage, cold_storage, cold_storage
engine handle: document-kv://req-local
engine segments: 3
disk shard: /tmp/cachet-quickstart-.../kvpacks/policy-handbook.kvpack
```

The bytes are not real model KV tensors. The example is intentionally small so
you can verify the Cachet data path without a GPU or model download.

## What The Example Does

1. Builds a `SourceDocument` with static text and two content chunks.
2. Builds a `DocumentKVRequest` selecting those chunks.
3. Uses `DocumentKVWorkflow.with_storage(...)` with memory and disk readers.
4. Uses `ToyKVGenerator` to emit deterministic fake `PackChunk` payloads.
5. Writes the chunks into a packed shard and records them in an in-memory
   manifest.
6. Materializes the selected chunks into a request payload.
7. Builds an `EngineReadyRequest` that a serving adapter could submit.

## Replace The Toy Parts

For real serving:

- replace `ToyKVGenerator` with a real KV generator for your model;
- use a real model `KVLayout`, such as `layout_for_model("qwen3:4b-instruct")`;
- use durable manifest storage instead of `InMemoryManifestStore`;
- use a vLLM, SGLang, or custom serving adapter to bind the handoff.

## Next

- Read [`concepts.md`](concepts.md) for the vocabulary.
- Read [`production.md`](production.md) for vLLM, SGLang, and managed-cloud notes.
- Read [`../benchmarks/README.md`](../benchmarks/) for the
  current benchmark summary.
