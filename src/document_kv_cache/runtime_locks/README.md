# Serving runtime locks

This directory contains the reviewed, hash-pinned Python dependency closure for
the vLLM serving runtime. The vLLM wheel itself is deliberately excluded from
the lock: it is installed by exact URI, either the official vLLM 0.30.0 CUDA
12.9 wheel or a wheel patched for the Cachet E5M2 contract by
`document_kv_cache.vllm_wheel_repack`.

`vllm-0.30.0-cu129-py311-manylinux_2_35.lock` targets the Databricks Runtime
CPython 3.11.11 interpreter on x86_64 Linux with glibc 2.35.

The controller-side semantic lock, which was never co-installed into the Linux
serving runtime, and the publication runners that consumed these locks during
the vLLM 0.30.0 campaign are preserved at the `vllm-0300-campaign-archive` tag.

## vLLM qualification lock

Byte-for-byte artifact replay means consuming the checked-in lock unchanged at
runtime, not recompiling or rewriting it during a job. Before installation,
verify that its SHA-256 is
`8b6f8618fff84070d44b35a2d6a54d5fc5831ecd8c375b5cbe9d847aeb3fae5d`, the
digest pinned by `document_kv_cache.serving_env`. Its exact four-line index
header is the sole runtime index authority: PyPI, the CUDA 12.9 PyTorch index,
and the two reviewed FlashInfer indexes. The engine-probe runner removes inherited
`PIP_*` options and disables every pip configuration file before creating its
virtualenv and before every install, check, and provenance-verifier
subprocess.

`uv --torch-backend cu129` selects and hash-locks the CUDA 12.9 PyTorch wheels,
but `--emit-index-url` does not emit that backend as a runtime index directive.
The unmodified compiler output therefore has SHA-256
`8cc92b1829969e9e1ff811b42fb5d8972fa80d4f6b2ca3e148af0baaae7a29e9` and
cannot install its existing `torch==2.13.0+cu129` pin with pip. The reviewed
post-compile transformer inserts exactly
`--extra-index-url https://download.pytorch.org/whl/cu129` after the PyPI line,
verifies both the pre- and post-transform digests, and leaves every requirement,
annotation, version, and distribution hash unchanged.

The generated comment at the top of the lock records the original compiler
invocation; its abbreviated Python version and implicit index strategy are
provenance, not the normative regeneration recipe.

The Cachet wheel pins `packaging==26.3` to match this serving closure.

Regeneration creates a new candidate vLLM lock. From the repository root, install
the reviewed compiler with `python -m pip install uv==0.11.6`, then run:

```sh
uv pip compile \
  src/document_kv_cache/runtime_locks/vllm-0.30.0-cu129-py311-manylinux_2_35.in \
  --output-file /tmp/vllm-0.30.0-cu129-py311-manylinux_2_35.compiled.lock \
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

Then replay the exact package-owned augmentation from the repository root:

```sh
PYTHONPATH=src python -c \
  'from pathlib import Path; from document_kv_cache.serving_env import augment_vllm_runtime_lock_indexes; source = Path("/tmp/vllm-0.30.0-cu129-py311-manylinux_2_35.compiled.lock"); target = Path("src/document_kv_cache/runtime_locks/vllm-0.30.0-cu129-py311-manylinux_2_35.lock"); target.write_bytes(augment_vllm_runtime_lock_indexes(source.read_bytes()))'
```

Any regenerated output requires review of the complete version-and-hash diff,
an updated lock digest, and fresh L4 and A10G GPU qualification before
publication jobs may run.
