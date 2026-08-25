# vLLM runtime locks

This directory contains the reviewed, hash-pinned Python dependency closure for
the publication runtime. The vLLM wheel itself is deliberately excluded from
the lock: it is built separately from the official vLLM 0.27.1 CUDA 12.9 wheel,
patched for the Cachet E5M2 contract, and installed by exact URI and SHA-256.

`vllm-0.27.1-cu129-py311-manylinux_2_35.lock` targets the Databricks Runtime
CPython 3.11.11 interpreter on x86_64 Linux with glibc 2.35.

Byte-for-byte artifact replay means consuming the checked-in lock unchanged,
not recompiling it. Before installation, verify that its SHA-256 is
`5788ee492a9a9ff48c8e1eae68cd0576fcec625263858129cc9dd918bcb856a6`, the
digest pinned by `document_kv_cache.serving_env`. The generated comment at the
top of the lock records the original compiler invocation; its abbreviated
Python version and implicit index strategy are provenance, not the normative
regeneration recipe.

Regeneration creates a new candidate lock. From the repository root, install
the reviewed compiler with `python -m pip install uv==0.11.6`, then run:

```sh
uv pip compile \
  src/document_kv_cache/runtime_locks/vllm-0.27.1-cu129-py311-manylinux_2_35.in \
  --output-file src/document_kv_cache/runtime_locks/vllm-0.27.1-cu129-py311-manylinux_2_35.lock \
  --python-version 3.11.11 \
  --python-platform x86_64-manylinux_2_35 \
  --only-binary :all: \
  --generate-hashes \
  --torch-backend cu129 \
  --index https://flashinfer.ai/whl/cu129 \
  --index https://flashinfer.ai/whl/ \
  --default-index https://pypi.org/simple \
  --index-strategy first-index \
  --emit-index-url \
  --emit-index-annotation \
  --no-emit-package vllm \
  --system-certs
```

Any regenerated output requires review of the complete version-and-hash diff,
an updated lock digest, and fresh L4 and A10G GPU qualification before
publication jobs may run.
