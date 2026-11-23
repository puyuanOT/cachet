# Current Dependency Freshness Evidence

This folder contains the current generated
`document_kv.dependency_freshness.v1` evidence for Cachet dependency policy.

The evidence distinguishes three dependency surfaces:

- Direct package metadata pins in `pyproject.toml`. These must be exact `==`
  pins and must match the latest stable package-index version supplied to the
  evidence generator.
- Isolated vLLM/SGLang serving-profile pins. These must be exact pins. A
  non-latest runtime pin is accepted only when the record carries an explicit
  compatibility or Databricks-validation hold reason.
- Resolved transitive drift from `poetry show --outdated --all`. Drift is
  accepted only when the record carries an explicit resolver-constraint reason.

Current direct pins are fresh: `poetry-core==2.5.0`, `packaging==26.3`,
`databricks-sdk==0.170.0`, and `pytest==9.2.0`.

Package-index observations were refreshed on 2026-11-23: PyPI for most
packages, the PyTorch CUDA 12.9 index for `torch`, `torchaudio`,
`torchvision`, and `torchcodec`, and the FlashInfer indexes for
`flashinfer-cubin` and `flashinfer-jit-cache`.

The vLLM profile is the reviewed `0.30.0+cu129` publication closure: its direct
CUDA/runtime packages are exact pins, while all transitive packages are bound
by the packaged 198-distribution hash lock. vLLM 0.34.0 is the latest release
and publishes no CUDA 12.9 wheel, so `vllm` and 24 closure packages carry
explicit holds until a vLLM runtime port is validated on L4 and A10G
Databricks hardware. The CUDA 12.9 `torch` family and `quack-kernels` are
current. The SGLang runtime pin stays on the Databricks-validated HiCache
provider profile.

`poetry show --outdated --all` reports only the dev-group tools `mypy` and
`ruff`, which their exact dev-group requirements hold; upgrading them adds
lint and type findings and is a separate lint-tooling change. The lock was
regenerated for this refresh, so every other locked package, including
`protobuf`, is at its latest release.

Files:

- `dependency-freshness-evidence.json`: generated freshness evidence record.

Regenerate from the repository root with:

```bash
python -m document_kv_cache.dependency_freshness \
  --pyproject pyproject.toml \
  --latest-version poetry-core=2.5.0 \
  --latest-version packaging=26.3 \
  --latest-version databricks-sdk=0.170.0 \
  --latest-version pytest=9.2.0 \
  --latest-version vllm=0.34.0 \
  --latest-version torch=2.13.0+cu129 \
  --latest-version torchaudio=2.11.0+cu129 \
  --latest-version torchvision=0.28.0+cu129 \
  --latest-version triton=3.9.0 \
  --latest-version numba=0.68.1 \
  --latest-version torchcodec=0.16.0+cu129 \
  --latest-version PyNvVideoCodec=2.2.4 \
  --latest-version flashinfer-python=0.7.5.post1 \
  --latest-version flashinfer-cubin=0.7.5.post1 \
  --latest-version flashinfer-jit-cache=0.7.5.post1+cu129 \
  --latest-version apache-tvm-ffi=0.1.16 \
  --latest-version tilelang=0.1.15 \
  --latest-version nvidia-cudnn-frontend=1.33.0 \
  --latest-version nvtx=0.2.17 \
  --latest-version fastsafetensors=0.5.0 \
  --latest-version instanttensor=0.2.3 \
  --latest-version nvidia-cutlass-dsl=4.8.1 \
  --latest-version quack-kernels=0.6.5 \
  --latest-version tokenspeed-mla=2.0.12 \
  --latest-version humming-kernels=0.1.20 \
  --latest-version transformers=5.24.0 \
  --latest-version huggingface-hub=2.4.0 \
  --latest-version bitsandbytes=0.50.2 \
  --latest-version accelerate=1.16.0 \
  --latest-version opencv-python-headless=5.1.0.95 \
  --latest-version tokenizers=0.23.3 \
  --latest-version numpy=2.5.5 \
  --latest-version fastapi=0.165.0 \
  --latest-version prometheus-fastapi-instrumentator=8.1.0 \
  --latest-version sglang=0.5.23 \
  --allow-runtime-pin 'vllm=Held at the reviewed vLLM 0.30.0+cu129 serving profile: the E5M2-patched wheel, packaged hash lock, and sealed L4/A10G Databricks benchmark evidence target 0.30.0, and vLLM 0.34.0 publishes no CUDA 12.9 wheel; upgrading requires a new runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'sglang=Pinned to the latest g6/L4 Databricks-validated Cachet SGLang HiCache provider profile; upgrading requires a fresh native/live Databricks validation.' \
  --allow-runtime-pin 'triton=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'numba=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'PyNvVideoCodec=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'flashinfer-python=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'flashinfer-cubin=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'flashinfer-jit-cache=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'apache-tvm-ffi=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'tilelang=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'nvidia-cudnn-frontend=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'nvtx=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'fastsafetensors=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'instanttensor=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'nvidia-cutlass-dsl=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'tokenspeed-mla=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'humming-kernels=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'transformers=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'huggingface-hub=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'bitsandbytes=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'accelerate=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'opencv-python-headless=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'tokenizers=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'numpy=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'fastapi=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --allow-runtime-pin 'prometheus-fastapi-instrumentator=Pinned by the reviewed vLLM 0.30.0+cu129 serving profile and its CUDA 12.9 hash lock; upgrade only with a vLLM runtime port and fresh L4 and A10G Databricks validation.' \
  --outdated-package mypy=2.2.0:2.4.1 \
  --outdated-package ruff=0.15.21:0.17.6 \
  --allow-transitive-outdated 'mypy=mypy==2.2.0 is an exact Poetry dev-group requirement, so the resolver keeps 2.2.0 even though PyPI has 2.4.1; upgrading adds type findings on the CI-checked files and is a separate lint-tooling change.' \
  --allow-transitive-outdated 'ruff=ruff==0.15.21 is an exact Poetry dev-group requirement, so the resolver keeps 0.15.21 even though PyPI has 0.17.6; upgrading adds lint findings on the CI-checked files and is a separate lint-tooling change.' \
  --output-json docs/release-ops/evidence/dependency-freshness/current/dependency-freshness-evidence.json
```

Do not commit package-index credentials, service tokens, raw resolver logs,
virtual environments, wheels, or local scratch output here.
