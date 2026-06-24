# Concepts

## Cachet In One Sentence

Cachet prepares reusable document-prefix KV payloads and the metadata needed to
hand those payloads to a serving engine.

## Glossary

| Term | Meaning |
| --- | --- |
| KV cache | Key/value attention tensors produced by a transformer while reading prompt tokens. Reusing them can avoid recomputing a stable prefix. |
| Document | A logical source text, such as a policy, article, ticket, contract, or benchmark passage. |
| Chunk | A selectable part of a document. Cachet supports static chunks and content chunks so requests can reuse only the context they need. |
| Manifest | The lookup table that maps model/document/chunk identities to packed KV payload byte ranges. |
| `.kvpack` | Cachet's packed file format for storing many chunk payloads in one shard instead of one file per chunk. |
| Materialization | Loading the selected chunk byte ranges from memory, disk, or other storage into the request payload shape. |
| Handoff | The metadata and bytes Cachet gives to a serving adapter so the adapter can bind the cached prefix to a request. |
| vLLM | A high-throughput LLM serving engine. Cachet includes thin vLLM adapter modules for native KV-transfer integration. |
| SGLang | A serving framework with HiCache support. Cachet includes thin SGLang adapter modules for HiCache-style handoffs. |
| Generator | Code that produces KV payload bytes for a document chunk. The local quickstart uses a fake generator; production uses a model-aware generator. |
| Layout | The model-specific KV geometry, dtype, block size, and storage layout a serving adapter needs to interpret payload bytes. |

## Data Flow

```text
SourceDocument
  -> SourceChunk records
  -> generator emits PackChunk payloads
  -> Cachet writes .kvpack shards
  -> manifest stores byte ranges
  -> DocumentKVRequest selects chunks
  -> materializer loads bytes
  -> EngineReadyRequest carries handoff metadata
```

## Cachet Does Not Own Decode

Cachet does not schedule requests, decode tokens, implement attention kernels,
or replace your serving engine. The serving engine owns runtime execution.
Cachet owns reusable document-prefix preparation and handoff.

## Stable Public API

Start with the Cachet-branded import surface:

```python
from cachet import SourceDocument, DocumentKVRequest, DocumentKVWorkflow
```

Advanced modules such as `cachet.engine_launch_config`,
`cachet.benchmark_handoffs`, `vllm_kv_injection`, and
`sglang_kv_injection` are available for integration work, but most users should
begin with the workflow objects above.

