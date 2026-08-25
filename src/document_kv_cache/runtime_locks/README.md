# Publication runtime locks

This directory contains the reviewed, hash-pinned Python dependency closure for
the publication runtime. The vLLM wheel itself is deliberately excluded from
the lock: it is built separately from the official vLLM 0.27.1 CUDA 12.9 wheel,
patched for the Cachet E5M2 contract, and installed by exact URI and SHA-256.

`vllm-0.27.1-cu129-py311-manylinux_2_35.lock` targets the Databricks Runtime
CPython 3.11.11 interpreter on x86_64 Linux with glibc 2.35.

`publication-latency-semantic-py311-macos-arm64.lock` is the complete
27-distribution controller-side environment used to rebuild and validate the
frozen latency handoff plan. It targets CPython 3.11.16 on
`aarch64-apple-darwin`. Its direct input and closed output are pinned as:

- `publication-latency-semantic-py311-macos-arm64.in`:
  `779938c5750a46931bb1a92eaadf09f83b12629542bb99a5ed2f100ec2a12034`
- `publication-latency-semantic-py311-macos-arm64.lock`:
  `8e26c54c74af9af63c5425e97581f3f9d1ecee00b28c7f151a607f673c14ccbb`

Replay consumes those checked-in lock bytes unchanged. The authority creates a
fresh private environment and installs the lock offline with
`--require-hashes`, `--only-binary :all:`, and `--no-deps`; it never resolves
an unhashed requirement or builds an sdist.

## Semantic-lock regeneration

The reviewed compiler identity is `uv 0.11.6 (65950801c 2026-04-09
aarch64-apple-darwin)`, executable SHA-256
`94151d6624054c3973829c82eb718db1afc55ef9fcee499cdd94bfb852fb99f9`.
The reviewed interpreter is
`/opt/homebrew/opt/python@3.11/bin/python3.11`, CPython 3.11.16. With no UV
index environment overrides, this is the complete normative command from the
repository root:

```sh
/Users/pliu/.local/bin/uv pip compile \
  src/document_kv_cache/runtime_locks/publication-latency-semantic-py311-macos-arm64.in \
  --output-file src/document_kv_cache/runtime_locks/publication-latency-semantic-py311-macos-arm64.lock \
  --python /opt/homebrew/opt/python@3.11/bin/python3.11 \
  --python-version 3.11.16 \
  --python-platform aarch64-apple-darwin \
  --only-binary :all: \
  --generate-hashes \
  --no-annotate \
  --no-header \
  --no-sources \
  --default-index https://pypi.org/simple \
  --index-strategy first-index \
  --no-python-downloads \
  --no-config \
  --system-certs
```

That command was independently replayed byte-for-byte to the pinned lock SHA
above. Regeneration creates only a candidate: any changed input or output
requires review of the complete version-and-hash closure, updated lock-byte and
installed-site closure pins in `publication_freeze.py`, and a fresh semantic
rebuild before source authority may issue.

## vLLM qualification lock

Byte-for-byte artifact replay means consuming the checked-in lock unchanged at
runtime, not recompiling or rewriting it during a job. Before installation,
verify that its SHA-256 is
`71c2c3e344ebdf1d8996adf2127a519328b6bad78a4eb7134c73e2a3f6115c44`, the
digest pinned by `document_kv_cache.serving_env`. Its exact four-line index
header is the sole runtime index authority: PyPI, the CUDA 12.9 PyTorch index,
and the two reviewed FlashInfer indexes. Package installation removes inherited
`PIP_*` options and disables every pip configuration file before invoking pip.
The standalone smoke path, engine-probe runner, generic Databricks bootstrap,
qualification bootstrap/sentinel worker, latency job/handoff/source-closure
runners, and full-score runner apply the same scrubbed environment to
virtualenv creation and every install, check, and provenance-verifier
subprocess.

`uv --torch-backend cu129` selects and hash-locks the CUDA 12.9 PyTorch wheels,
but `--emit-index-url` does not emit that backend as a runtime index directive.
The unmodified compiler output therefore has SHA-256
`5788ee492a9a9ff48c8e1eae68cd0576fcec625263858129cc9dd918bcb856a6` and
cannot install its existing `torch==2.13.0+cu129` pin with pip. The reviewed
post-compile transformer inserts exactly
`--extra-index-url https://download.pytorch.org/whl/cu129` after the PyPI line,
verifies both the pre- and post-transform digests, and leaves every requirement,
annotation, version, and distribution hash unchanged.

The generated comment at the top of the lock records the original compiler
invocation; its abbreviated Python version and implicit index strategy are
provenance, not the normative regeneration recipe.

The Cachet wheel pins `packaging==26.3` to match this serving closure. The
separate `publication-latency-semantic-py311-macos-arm64.in` and `.lock` remain
at `packaging==26.2`: that macOS-only tokenizer/semantic validation tool is an
isolated build input and is never co-installed into the Linux serving runtime.

Regeneration creates a new candidate vLLM lock. From the repository root, install
the reviewed compiler with `python -m pip install uv==0.11.6`, then run:

```sh
uv pip compile \
  src/document_kv_cache/runtime_locks/vllm-0.27.1-cu129-py311-manylinux_2_35.in \
  --output-file /tmp/vllm-0.27.1-cu129-py311-manylinux_2_35.compiled.lock \
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
  'from pathlib import Path; from document_kv_cache.serving_env import augment_vllm_runtime_lock_indexes; source = Path("/tmp/vllm-0.27.1-cu129-py311-manylinux_2_35.compiled.lock"); target = Path("src/document_kv_cache/runtime_locks/vllm-0.27.1-cu129-py311-manylinux_2_35.lock"); target.write_bytes(augment_vllm_runtime_lock_indexes(source.read_bytes()))'
```

Any regenerated output requires review of the complete version-and-hash diff,
an updated lock digest, and fresh L4 and A10G GPU qualification before
publication jobs may run.
