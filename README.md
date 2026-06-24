# Cachet

Cachet is a Python package for preparing reusable document KV-cache payloads
for long-context LLM serving. It helps you turn stable document context into
materialized bytes and handoff metadata that a serving engine can bind at
request time.

Cachet does **not** replace vLLM, SGLang, Transformers, or your inference
server. It owns the work around the engine: document manifests, packed KV shard
storage, memory/disk cache tiers, materialization, and engine handoff records.

## Who It Is For

Use Cachet when you:

- repeatedly answer questions over long, mostly stable documents;
- want to precompute or reuse document prefix KV outside the hot request path;
- need a Python API that can hand materialized document KV to vLLM, SGLang, or a
  custom adapter;
- want benchmarkable, auditable comparisons between cached-prefix serving and
  ordinary no-cache prefill.

Cachet is probably not the right first tool when you:

- only have short prompts;
- need one-off generation with no repeated document context;
- want Cachet to run model decode or scheduling itself;
- are looking for a vector database or retrieval system rather than KV-cache
  orchestration.

## Flow

```text
source document
  -> chunks + manifest
  -> packed KV payloads
  -> materialized request payload
  -> vLLM / SGLang / custom engine handoff
```

## Install

```bash
pip install cachet-kv
```

Cachet supports Python `>=3.11,<4.0`. The base package is intentionally small.
Serving-engine runtimes such as vLLM, SGLang, Transformers, and cloud-specific
libraries are installed separately by the environment that needs them.

## 10-Minute Local Quickstart

The local quickstart uses a tiny fake KV generator. It does not need a GPU,
cloud account, model download, vLLM, or SGLang. It proves the Cachet plumbing:
document -> packed payload -> materialized request -> engine-ready handoff.

After installing the package:

```bash
python -m cachet.quickstart_local
```

From a source checkout, the repo example delegates to the same packaged module:

```bash
python examples/quickstart_local.py
```

Expected output looks like:

```text
generated chunks: 4
materialized bytes: 340
engine handle: document-kv://req-local
disk shard: ...
```

The example writes a local `.kvpack` shard into a temporary directory, also
shows a memory-backed shard, and builds an `EngineReadyRequest`. The bytes are
fake; a production integration swaps in a real KV generator such as a
Transformers-backed generator and a real serving-engine adapter.

## Minimal API Shape

```python
from cachet import (
    CacheBuildConfig,
    DocumentKVRequest,
    DocumentKVWorkflow,
    InMemoryManifestStore,
    SourceDocument,
    layout_for_model,
)

manifest = InMemoryManifestStore()
workflow = DocumentKVWorkflow.with_storage(manifest=manifest)
layout = layout_for_model("qwen3:4b-instruct")
prompt_template_version = "v1"

document = SourceDocument.from_text(
    document_id="handbook",
    text="Long document text that will be cached...",
)

request = DocumentKVRequest.for_text_document(
    request_id="req-1",
    task_id="qa",
    model_id=layout.model_id,
    lora_id=layout.lora_id,
    prompt_template_version=prompt_template_version,
    document_id=document.document_id,
)

config = CacheBuildConfig(
    model_id=layout.model_id,
    lora_id=layout.lora_id,
    prompt_template_version=prompt_template_version,
    dtype=layout.dtype,
    layout_version=layout.layout_version,
    storage_layout=layout.storage_layout,
)
```

See [`examples/quickstart_local.py`](examples/quickstart_local.py) and the
packaged `cachet.quickstart_local` module for the complete runnable version,
including the toy generator and local materialization step.

## Where To Go Next

| Audience | Start here |
| --- | --- |
| New users | [`docs/getting-started.md`](docs/getting-started.md) |
| Concepts and vocabulary | [`docs/concepts.md`](docs/concepts.md) |
| Local runnable example | [`examples/quickstart_local.py`](examples/quickstart_local.py) |
| Production serving | [`docs/production.md`](docs/production.md) |
| vLLM/SGLang adapter maintainers | [`docs/native-engine-integration.md`](docs/native-engine-integration.md) |
| Benchmarks | [`benchmarks/current/README.md`](benchmarks/current/) |
| Repository map | [`docs/repo-map.md`](docs/repo-map.md) |
| Maintainer operations | [`docs/release-ops/README.md`](docs/release-ops/README.md) |

## Stable User Surface

Prefer the Cachet-branded Python API:

```python
from cachet import SourceDocument, DocumentKVRequest, DocumentKVWorkflow
```

The stable public import is `cachet`. The distribution name is `cachet-kv`
because the exact `cachet` package name on PyPI is owned by an unrelated
project.

Most users start with the Python API. The packaged local example is:

```bash
python -m cachet.quickstart_local
```

The first serving-oriented CLI to learn is:

```bash
cachet-engine-launch-config --help
```

Maintainer-only CLIs are documented under
[`docs/release-ops/`](docs/release-ops/README.md), not in the beginner path.

## Current Status

Cachet is alpha software. The current human-readable benchmark summary lives in
[`benchmarks/current/README.md`](benchmarks/current/). You do not need cloud
infrastructure to try the local quickstart.

## Contributing

External issues and pull requests are welcome. Start with
[`CONTRIBUTING.md`](CONTRIBUTING.md), which describes the public contribution
path. Maintainer-only release gates live under
[`docs/release-ops/`](docs/release-ops/README.md).

## License

Cachet is distributed under the Apache License 2.0. See [`LICENSE`](LICENSE).
