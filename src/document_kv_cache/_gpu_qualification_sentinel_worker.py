"""GPU-only sentinel worker for the closed vLLM 0.27.1 qualification plan.

This module is deliberately not imported by the Databricks payload renderer.
It runs inside the hash-locked Python 3.11/cu129 environment created by
``gpu_qualification_sentinels``.  Every dispatch target below is package owned;
there is no factory flag and no path that promotes caller-provided JSON to
qualification evidence.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS,
    GPU_QUALIFICATION_A10G_MAX_MODEL_LEN,
    GPU_QUALIFICATION_A10G_REQUIRED_KV_CAPACITY_TOKENS,
    GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256,
    GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
    GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256,
    GPU_QUALIFICATION_MAX_E5M2_BF16_ERROR,
    GPU_QUALIFICATION_MAX_LOGIT_DRIFT,
    GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES,
    GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND,
    GPU_QUALIFICATION_MODEL_ID,
    GPU_QUALIFICATION_MODEL_LAYER_COUNT,
    GPU_QUALIFICATION_MODEL_REVISION,
    GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS,
    GPU_QUALIFICATION_REQUEST_PARALLELISM,
    GPU_QUALIFICATION_INPUT_CONTEXT_TOKENS,
    GPU_QUALIFICATION_INPUT_DATASETS,
    GPU_QUALIFICATION_INPUT_ROWS_PER_DATASET_BUCKET,
    GPU_QUALIFICATION_MAX_MODEL_LEN,
    GPU_QUALIFICATION_THROUGHPUT_BUCKETS,
    GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET,
    GPU_QUALIFICATION_VLLM_VERSION,
    canonical_gpu_qualification_json,
)
from document_kv_cache.serving_env import (
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
    VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_LOCK_SHA256,
)


_RUNTIME_LOCK_ATTESTATION_ENV: Final = (
    "CACHET_GPU_QUALIFICATION_RUNTIME_LOCK_ATTESTATION"
)
_CORE_VERSIONS: Final = {
    "bitsandbytes": "0.49.2",
    "flashinfer-cubin": "0.6.16.post3",
    "flashinfer-jit-cache": "0.6.16.post3+cu129",
    "flashinfer-python": "0.6.16.post3",
    "torch": "2.13.0+cu129",
    "torchaudio": "2.11.0+cu129",
    "torchcodec": "0.16.0+cu129",
    "torchvision": "0.28.0+cu129",
    "triton": "3.7.1",
}
_PATCH_MEMBER_SHA256: Final = {
    "vllm/model_executor/layers/attention/attention.py": (
        "5735acfb390cf344caeec950c2f286344bcd84721ce287e0a56701f2a18bc839"
    ),
    "vllm/v1/attention/backends/triton_attn.py": (
        "4dae0ff6c4ee8f11c1f195151a11673d595d457c413032e7bae7550913f94390"
    ),
    "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py": (
        "0682ca7bc56edf7cea5419188a81c78510b54192471472b160aa447ac0ceeb08"
    ),
}
_CONNECTOR_BASE_RELATIVE_PATH: Final = (
    "vllm/distributed/kv_transfer/kv_connector/v1/base.py"
)
_EXPECTED_SENTINELS: Final = frozenset(
    {
        "forced_triton_runtime_handoff",
        "packed_page_raw_byte_roundtrip",
        "matched_token_contract_and_determinism",
        "l4_32k_c4_gmu_sweep",
        "a10g_16k_c4_capacity",
        "generation_throughput_with_writes",
        "auto_backend_diagnostic",
    }
)


def execute_planned_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Run exactly one plan-selected sentinel and return measured fields."""

    sentinel = _required_string(planned_job, "sentinel")
    if sentinel not in _EXPECTED_SENTINELS:
        raise ValueError(f"unsupported GPU qualification sentinel: {sentinel!r}")
    _attest_gpu_target(planned_job)
    work_dir.mkdir(parents=True, exist_ok=False)

    dispatch: dict[str, Callable[..., dict[str, Any]]] = {
        "forced_triton_runtime_handoff": _runtime_handoff_sentinel,
        "packed_page_raw_byte_roundtrip": _packed_roundtrip_sentinel,
        "matched_token_contract_and_determinism": _matched_token_sentinel,
        "l4_32k_c4_gmu_sweep": _gmu_sentinel,
        "a10g_16k_c4_capacity": _a10g_capacity_sentinel,
        "generation_throughput_with_writes": _throughput_sentinel,
        "auto_backend_diagnostic": _auto_backend_sentinel,
    }
    measured = dispatch[sentinel](
        plan_record=plan_record,
        planned_job=planned_job,
        input_bundle=input_bundle,
        work_dir=work_dir,
    )
    return _finite_json_object(measured, "sentinel measurements")


def _attest_gpu_target(planned_job: Mapping[str, Any]) -> None:
    torch = _torch()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU qualification requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    expected_name = _required_string(planned_job, "gpu")
    expected_capability = _required_string(planned_job, "compute_capability")
    observed_capability = f"{properties.major}.{properties.minor}"
    if properties.name != expected_name:
        raise RuntimeError(
            f"planned GPU {expected_name!r}, observed {properties.name!r}"
        )
    if observed_capability != expected_capability:
        raise RuntimeError(
            "planned compute capability "
            f"{expected_capability}, observed {observed_capability}"
        )


def _triton_e5m2_probe(work_dir: Path) -> dict[str, Any]:
    """Compile and launch vLLM's two publication kernels on raw E5M2 pages."""

    torch = _torch()
    cache_dir = work_dir / "triton-cache-miss"
    cache_dir.mkdir(parents=True, exist_ok=False)
    if any(cache_dir.iterdir()):
        raise RuntimeError("fresh Triton cache directory was not empty")
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)

    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # type: ignore[import-not-found]
        triton_reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.triton_unified_attention import (  # type: ignore[import-not-found]
        unified_attention,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode  # type: ignore[import-not-found]

    torch.manual_seed(20260824)
    token_count, block_size, num_heads, head_size = 8, 16, 8, 128
    key = (
        torch.randint(
            -4,
            5,
            (token_count, num_heads, head_size),
            device="cuda",
            dtype=torch.int16,
        ).to(torch.bfloat16)
        / 4
    )
    value = torch.flip(key, dims=(-1,))
    key_raw = torch.full(
        (1, block_size, num_heads, head_size),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )
    value_raw = torch.full_like(key_raw, 0xA5)
    slots = torch.arange(token_count, device="cuda", dtype=torch.long)
    scale = torch.ones((), device="cuda", dtype=torch.float32)

    triton_reshape_and_cache_flash(
        key,
        value,
        key_raw,
        value_raw,
        slots,
        "fp8_e5m2",
        scale,
        scale,
    )
    torch.cuda.synchronize()
    compiled_after_reshape = tuple(
        path for path in cache_dir.rglob("*") if path.is_file()
    )
    if not compiled_after_reshape:
        raise RuntimeError("reshape-and-cache did not populate the fresh Triton cache")

    query = key[-1:].clone()
    output = torch.empty_like(query)
    key_cache = key_raw.view(torch.float8_e5m2)
    value_cache = value_raw.view(torch.float8_e5m2)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor(
        [token_count], device="cuda", dtype=torch.int32
    )
    block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    descale = torch.ones((1, num_heads), device="cuda", dtype=torch.float32)
    softmax_scale = head_size**-0.5
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=sequence_lengths,
        max_seqlen_k=token_count,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=descale,
        v_descale=descale,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    torch.cuda.synchronize()
    compiled_after_attention = tuple(
        path for path in cache_dir.rglob("*") if path.is_file()
    )
    if len(compiled_after_attention) <= len(compiled_after_reshape):
        raise RuntimeError("unified attention did not compile into the fresh cache")

    decoded_key = key_cache[0, :token_count].to(torch.float32)
    decoded_value = value_cache[0, :token_count].to(torch.float32)
    scores = torch.einsum(
        "qhd,thd->qht", query.float(), decoded_key
    ) * softmax_scale
    reference = torch.einsum(
        "qht,thd->qhd", torch.softmax(scores, dim=-1), decoded_value
    )
    max_error = float((output.float() - reference).abs().max().item())
    if not math.isfinite(max_error) or max_error > GPU_QUALIFICATION_MAX_E5M2_BF16_ERROR:
        raise RuntimeError(f"E5M2 Triton probe exceeded BF16 error bound: {max_error}")
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("E5M2 Triton probe produced non-finite output")

    # Repeat both launches after compilation.  This demonstrates that the same
    # call sites execute from the populated cache and gives an observed launch
    # count rather than inferring success from filesystem artifacts alone.
    triton_reshape_and_cache_flash(
        key,
        value,
        key_raw,
        value_raw,
        slots,
        "fp8_e5m2",
        scale,
        scale,
    )
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=sequence_lengths,
        max_seqlen_k=token_count,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=descale,
        v_descale=descale,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    torch.cuda.synchronize()
    return {
        "bf16_reference_max_abs_error": max_error,
        "finite_logits": True,
        "triton_cache_miss_compile": True,
        "triton_compile_count": 2,
        "triton_compiled_kernel_names": [
            "triton_reshape_and_cache_flash",
            "triton_unified_attention",
        ],
        "triton_kernel_launch_count": 4,
    }


def _packed_roundtrip_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
) -> dict[str, Any]:
    del plan_record, planned_job, input_bundle
    torch = _torch()
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode
    from vllm_kv_injection.paged_kv_copy import inject_kv_cache_layer

    cache_dir = work_dir / "triton-packed-cache"
    cache_dir.mkdir()
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    cases: list[dict[str, Any]] = []
    token_count, block_size, heads, head_size = 8, 16, 8, 128
    for layout_index, payload_layout in enumerate(("NHD", "HND")):
        values = (
            (
                torch.arange(
                    token_count * 2 * heads * head_size,
                    device="cuda",
                    dtype=torch.int64,
                )
                % 9
            ).to(torch.bfloat16)
            - 4
        ) / 4
        values = values.reshape(token_count, 2, heads, head_size)
        encoded = values.to(torch.float8_e5m2).view(torch.uint8)
        if payload_layout == "NHD":
            backing = torch.empty(
                (token_count, 2, heads, head_size * 2),
                device="cuda",
                dtype=torch.uint8,
            )
            backing[..., ::2] = encoded
            source = backing[..., ::2]
        else:
            backing = encoded.permute(1, 2, 0, 3).contiguous()
            source = backing.permute(2, 0, 1, 3)
        if source.is_contiguous():
            raise RuntimeError(f"{payload_layout} source did not exercise strides")

        destination = torch.full(
            (1, heads, block_size, 2 * head_size),
            0xA5,
            device="cuda",
            dtype=torch.uint8,
        )
        slots = torch.arange(token_count, device="cuda", dtype=torch.long)
        inject_kv_cache_layer(
            destination,
            source,
            slots,
            block_size=block_size,
        )
        torch.cuda.synchronize()
        read_key = destination[0, :, :token_count, :head_size].permute(1, 0, 2)
        read_value = destination[0, :, :token_count, head_size:].permute(1, 0, 2)
        read_raw = torch.stack((read_key, read_value), dim=1)
        mismatch_count = int((read_raw != source).sum().item())
        untouched = destination[0, :, token_count:]
        untouched_mismatches = int((untouched != 0xA5).sum().item())
        written_bytes = source.detach().cpu().contiguous().numpy().tobytes()
        read_bytes = read_raw.detach().cpu().contiguous().numpy().tobytes()

        snapshot = destination.clone()
        bad_slots = slots.clone()
        bad_slots[-1] = -1
        try:
            inject_kv_cache_layer(
                destination,
                source,
                bad_slots,
                block_size=block_size,
            )
        except ValueError:
            negative_guard = bool(torch.equal(snapshot, destination))
        else:
            negative_guard = False
        if not negative_guard:
            raise RuntimeError("negative slot guard failed or mutated the page")

        query = values[-1, 0].unsqueeze(0)
        output = torch.empty_like(query)
        key_cache = (
            destination[..., :head_size]
            .permute(0, 2, 1, 3)
            .view(torch.float8_e5m2)
        )
        value_cache = (
            destination[..., head_size:]
            .permute(0, 2, 1, 3)
            .view(torch.float8_e5m2)
        )
        descale = torch.ones((1, heads), device="cuda", dtype=torch.float32)
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            max_seqlen_q=1,
            seqused_k=torch.tensor(
                [token_count], device="cuda", dtype=torch.int32
            ),
            max_seqlen_k=token_count,
            softmax_scale=head_size**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=torch.tensor([[0]], device="cuda", dtype=torch.int32),
            softcap=0.0,
            q_descale=None,
            k_descale=descale,
            v_descale=descale,
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        )
        torch.cuda.synchronize()
        decoded_key = values[:, 0].to(torch.float8_e5m2).float()
        decoded_value = values[:, 1].to(torch.float8_e5m2).float()
        scores = torch.einsum(
            "qhd,thd->qht", query.float(), decoded_key
        ) * (head_size**-0.5)
        reference = torch.einsum(
            "qht,thd->qhd", torch.softmax(scores, dim=-1), decoded_value
        )
        error = float((output.float() - reference).abs().max().item())
        if error > GPU_QUALIFICATION_MAX_E5M2_BF16_ERROR:
            raise RuntimeError(f"packed {payload_layout} attention error {error}")
        cases.append(
            {
                "bf16_reference_max_abs_error": error,
                "bf16_reference_scope": "attention_output",
                "cache_page_layout": "B_H_N_2D",
                "cache_page_shape": ["B", "H", "N", "2D"],
                "input_value_max": float(values.max().item()),
                "input_value_min": float(values.min().item()),
                "negative_slot_guard_passed": negative_guard,
                "noncontiguous_stride_passed": not source.is_contiguous(),
                "partial_slot_guard_passed": untouched_mismatches == 0,
                "payload_layout": payload_layout,
                "query_dtype": "bfloat16",
                "raw_byte_mismatch_count": mismatch_count,
                "raw_bytes_written": len(written_bytes),
                "read_raw_sha256": sha256(read_bytes).hexdigest(),
                "untouched_guard_mismatch_count": untouched_mismatches,
                "written_raw_sha256": sha256(written_bytes).hexdigest(),
            }
        )
        if layout_index == 0 and not any(cache_dir.rglob("*")):
            raise RuntimeError("packed attention did not populate the Triton cache")
    return {
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "cases": cases,
        "triton_compile_count": 1,
        "triton_kernel_launch_count": len(cases),
    }


def _runtime_handoff_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
) -> dict[str, Any]:
    del input_bundle
    torch = _torch()
    probe = _triton_e5m2_probe(work_dir)
    handoff_counts = _exercise_all_layer_handoff(work_dir)
    installed = {
        distribution: importlib.metadata.version(distribution)
        for distribution in _CORE_VERSIONS
    }
    if installed != _CORE_VERSIONS:
        raise RuntimeError(f"locked runtime distributions differ: {installed!r}")
    runtime_lock_attestation = _runtime_lock_attestation()
    if importlib.metadata.version("vllm") != GPU_QUALIFICATION_VLLM_VERSION:
        raise RuntimeError("installed vLLM version differs from the qualification pin")
    patch_hashes = _installed_vllm_member_hashes(_PATCH_MEMBER_SHA256)
    if patch_hashes != _PATCH_MEMBER_SHA256:
        raise RuntimeError("installed vLLM patch member hashes differ")
    connector_hash = _installed_vllm_member_hashes(
        {_CONNECTOR_BASE_RELATIVE_PATH: GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256}
    )[_CONNECTOR_BASE_RELATIVE_PATH]
    direct_url_matches = _direct_url_matches_patched_wheel()
    if not direct_url_matches:
        raise RuntimeError("vLLM PEP 610 origin does not match the patched wheel")
    pip_check_ok = _pip_check_ok()
    if not pip_check_ok:
        raise RuntimeError("pip check failed in the isolated runtime")
    native_count, unresolved_count = _native_shared_object_resolution()
    if unresolved_count:
        raise RuntimeError("the isolated runtime has unresolved native objects")
    libcudart_majors = _libcudart_major_versions()
    if libcudart_majors != [12]:
        raise RuntimeError(f"unexpected libcudart majors: {libcudart_majors!r}")
    python_version = platform.python_version()
    glibc_version = platform.libc_ver()[1]
    system_cuda = _system_cuda_version()
    if (python_version, glibc_version, system_cuda) != ("3.11.11", "2.35", "12.1"):
        raise RuntimeError(
            "platform closure mismatch: "
            f"python={python_version}, glibc={glibc_version}, cuda={system_cuda}"
        )
    capability = torch.cuda.get_device_capability(0)
    software_path = capability < (8, 9)
    if (
        _required_string(planned_job, "hardware_id") == "aws-g5-a10g"
        and not software_path
    ):
        raise RuntimeError("A10G did not select the E5M2 software-capability path")
    runtime_contract = _mapping(plan_record.get("runtime_contract"), "runtime_contract")
    weight_attestation = _weight_quantizer_attestation()
    return {
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "compute_dtype": "bfloat16",
        "connector_source_sha256": connector_hash,
        "direct_url_matches_patched_wheel": direct_url_matches,
        "driver_cuda_compatibility_ok": torch.cuda.is_available(),
        "e5m2_software_path_exercised": software_path,
        "finite_logits": probe["finite_logits"],
        **handoff_counts,
        "installed_core_distribution_versions": installed,
        "installed_connector_base_py_sha256": connector_hash,
        "installed_patch_member_sha256": patch_hashes,
        "libcudart_major_versions": libcudart_majors,
        "libcudart_so_12_present": 12 in libcudart_majors,
        "libcudart_so_13_present": 13 in libcudart_majors,
        "model_id": runtime_contract["model_id"],
        "model_revision": runtime_contract["model_revision"],
        "native_shared_object_count": native_count,
        "pip_check_ok": pip_check_ok,
        "python_version": python_version,
        "query_dtype": "bfloat16",
        "runtime_kv_dtype": "fp8_e5m2",
        "runtime_kv_bits": 8,
        "runtime_lock_attestation": runtime_lock_attestation,
        "runtime_lock_verifier_ok": runtime_lock_attestation["ok"] is True,
        "site_packages_read_only": _site_packages_read_only(),
        "strict_direct_url_verifier_ok": direct_url_matches,
        "system_cuda_version": system_cuda,
        "glibc_version": glibc_version,
        "triton_cache_miss_compile": probe["triton_cache_miss_compile"],
        "triton_compile_count": probe["triton_compile_count"],
        "triton_compiled_kernel_names": probe["triton_compiled_kernel_names"],
        "triton_kernel_launch_count": probe["triton_kernel_launch_count"],
        "unresolved_native_shared_object_count": unresolved_count,
        "weight_bits": 4,
        "weight_quantization": "bitsandbytes",
        "trust_remote_code": False,
        "weight_quantizer_attestation": weight_attestation,
    }


def _runtime_lock_attestation() -> dict[str, Any]:
    raw = os.environ.get(_RUNTIME_LOCK_ATTESTATION_ENV)
    if not raw:
        raise RuntimeError("full runtime-lock attestation is unavailable")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("full runtime-lock attestation is invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "locked_distribution_count",
        "ok",
        "runtime_lock_sha256",
        "unexpected_distributions",
        "vllm_direct_url",
        "vllm_package_version",
        "vllm_wheel_sha256",
    }:
        raise RuntimeError("full runtime-lock attestation has an open schema")
    expected = {
        "locked_distribution_count": VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
        "ok": True,
        "runtime_lock_sha256": VLLM_RUNTIME_LOCK_SHA256,
        "unexpected_distributions": [],
        "vllm_package_version": GPU_QUALIFICATION_VLLM_VERSION,
        "vllm_wheel_sha256": os.environ.get(VLLM_PATCHED_WHEEL_SHA256_ENV),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"full runtime-lock attestation {key} differs")
    direct_url = value.get("vllm_direct_url")
    if not isinstance(direct_url, str) or not direct_url.startswith("file:"):
        raise RuntimeError("full runtime-lock attestation has no local vLLM origin")
    return dict(value)


def _exercise_all_layer_handoff(work_dir: Path) -> dict[str, Any]:
    """Persist, load, and inject one raw E5M2 payload across all 36 layers."""

    torch = _torch()
    from document_kv_cache.engine_protocol import KVStorageLayout
    from document_kv_cache.kvpack import PackChunk, write_kvpack
    from document_kv_cache.models import DocumentChunkType, KVCacheKey
    from document_kv_cache.storage import DiskRangeReader
    from vllm_kv_injection.paged_kv_copy import inject_kv_cache_layer

    tokens, heads, head_size, block_size = 4, 8, 128, 16
    encoded = (
        (
            torch.arange(
                tokens
                * GPU_QUALIFICATION_MODEL_LAYER_COUNT
                * 2
                * heads
                * head_size,
                device="cuda",
                dtype=torch.int64,
            )
            % 9
        ).to(torch.bfloat16)
        - 4
    ) / 4
    encoded = (
        encoded.reshape(
            tokens,
            GPU_QUALIFICATION_MODEL_LAYER_COUNT,
            2,
            heads,
            head_size,
        )
        .to(torch.float8_e5m2)
        .view(torch.uint8)
    )
    payload = encoded.detach().cpu().contiguous().numpy().tobytes()
    key = KVCacheKey.for_document(
        model_id=GPU_QUALIFICATION_MODEL_ID,
        lora_id="base",
        prompt_template_version="gpu-qualification-v1",
        document_id="all-layer-synthetic",
        chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
        chunk_id="payload",
        content_hash=sha256(payload).hexdigest(),
    )
    path = work_dir / "all-layer-handoff.kvpack"
    refs = write_kvpack(
        path,
        (
            PackChunk(
                key=key,
                payload=payload,
                token_count=tokens,
                dtype="fp8_e5m2",
                layout_version="qwen3-4b-v1",
                storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE,
            ),
        ),
    )
    if len(refs) != 1:
        raise RuntimeError("all-layer handoff did not produce exactly one ref")
    loaded = DiskRangeReader().read(refs[0])
    if loaded != payload:
        raise RuntimeError("durably loaded all-layer handoff bytes differ")
    loaded_tensor = (
        torch.frombuffer(bytearray(loaded), dtype=torch.uint8)
        .reshape(
            tokens,
            GPU_QUALIFICATION_MODEL_LAYER_COUNT,
            2,
            heads,
            head_size,
        )
        .to("cuda")
    )
    slots = torch.arange(tokens, device="cuda", dtype=torch.long)
    injected = 0
    for layer_index in range(GPU_QUALIFICATION_MODEL_LAYER_COUNT):
        destination = torch.zeros(
            (1, heads, block_size, head_size * 2),
            device="cuda",
            dtype=torch.uint8,
        )
        source = loaded_tensor[:, layer_index]
        inject_kv_cache_layer(
            destination,
            source,
            slots,
            block_size=block_size,
            validate=layer_index == 0,
        )
        read = torch.stack(
            (
                destination[0, :, :tokens, :head_size].permute(1, 0, 2),
                destination[0, :, :tokens, head_size:].permute(1, 0, 2),
            ),
            dim=1,
        )
        if not torch.equal(read, source):
            raise RuntimeError(f"all-layer injection differed at layer {layer_index}")
        injected += 1
    torch.cuda.synchronize()
    return {
        "handoff_injected": injected == GPU_QUALIFICATION_MODEL_LAYER_COUNT,
        "handoff_injected_layer_count": injected,
        "handoff_kv_bits": 8,
        "handoff_kv_dtype": "fp8_e5m2",
        "handoff_loaded": True,
        "handoff_loaded_layer_count": GPU_QUALIFICATION_MODEL_LAYER_COUNT,
        "handoff_written": path.is_file(),
        "handoff_written_layer_count": GPU_QUALIFICATION_MODEL_LAYER_COUNT,
    }


def _installed_vllm_member_hashes(
    expected: Mapping[str, str],
) -> dict[str, str]:
    distribution = importlib.metadata.distribution("vllm")
    root = Path(str(distribution.locate_file("")))
    observed: dict[str, str] = {}
    for relative_path in expected:
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"installed vLLM member is missing: {relative_path}")
        observed[relative_path] = _file_sha256(path)
    return observed


def _direct_url_matches_patched_wheel() -> bool:
    wheel_uri = os.environ.get(VLLM_PATCHED_WHEEL_URI_ENV)
    expected_sha256 = os.environ.get(VLLM_PATCHED_WHEEL_SHA256_ENV)
    if not wheel_uri or not expected_sha256:
        return False
    if _file_sha256(Path(wheel_uri)) != expected_sha256:
        return False
    distribution = importlib.metadata.distribution("vllm")
    text = distribution.read_text("direct_url.json")
    if text is None:
        return False
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return False
    url = record.get("url") if isinstance(record, Mapping) else None
    if not isinstance(url, str) or not url.startswith("file:"):
        return False
    from urllib.parse import unquote, urlsplit

    parsed = urlsplit(url)
    return Path(unquote(parsed.path)).resolve() == Path(wheel_uri).resolve()


def _pip_check_ok() -> bool:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed.returncode == 0


def _native_shared_object_resolution() -> tuple[int, int]:
    roots = {
        Path(str(importlib.metadata.distribution(name).locate_file("")))
        for name in ("vllm", "torch", "bitsandbytes", "triton")
    }
    objects = sorted(
        {
            path.resolve()
            for root in roots
            for pattern in ("*.so", "*.so.*")
            for path in root.rglob(pattern)
            if path.is_file() and not path.is_symlink()
        }
    )
    if not objects:
        raise RuntimeError("no native runtime shared objects were found")
    unresolved = 0
    for path in objects:
        completed = subprocess.run(
            ["ldd", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Some extension objects are static/non-dynamic; ldd returns 1 for
        # those, which is not an unresolved dependency.  Only concrete
        # ``=> not found`` bindings count as unresolved.
        unresolved += sum(
            1 for line in completed.stdout.splitlines() if "=> not found" in line
        )
    return len(objects), unresolved


def _libcudart_major_versions() -> list[int]:
    roots = {
        Path(str(importlib.metadata.distribution(name).locate_file("")))
        for name in ("torch", "nvidia-cuda-runtime-cu12")
        if _distribution_exists(name)
    }
    majors: set[int] = set()
    pattern = re.compile(r"libcudart\.so\.(\d+)")
    for root in roots:
        for path in root.rglob("libcudart.so*"):
            match = pattern.search(path.name)
            if match:
                majors.add(int(match.group(1)))
    return sorted(majors)


def _system_cuda_version() -> str:
    version_json = Path("/usr/local/cuda/version.json")
    if version_json.is_file():
        record = json.loads(version_json.read_text(encoding="utf-8"))
        cuda = record.get("cuda") if isinstance(record, Mapping) else None
        version = cuda.get("version") if isinstance(cuda, Mapping) else None
        if isinstance(version, str):
            return ".".join(version.split(".")[:2])
    completed = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r"release\s+(\d+\.\d+)", completed.stdout)
    if match is None:
        raise RuntimeError("could not attest the system CUDA toolkit version")
    return match.group(1)


def _site_packages_read_only() -> bool:
    import site
    import stat

    for raw_path in site.getsitepackages():
        path = Path(raw_path)
        if not path.is_dir() or path.is_symlink():
            return False
        for child in (path, *path.rglob("*")):
            if child.is_symlink():
                continue
            if child.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                return False
    return True


def _weight_quantizer_attestation() -> dict[str, Any]:
    """Exercise the pinned NF4/double-quant call and inspect its output state."""

    torch = _torch()
    import bitsandbytes.functional as bnb_functional  # type: ignore[import-not-found]
    from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

    loader_hash = _installed_vllm_member_hashes(
        {
            "vllm/model_executor/model_loader/bitsandbytes_loader.py": (
                GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
            )
        }
    )["vllm/model_executor/model_loader/bitsandbytes_loader.py"]
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
    )
    source = torch.linspace(
        -1.0,
        1.0,
        steps=128 * 128,
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(128, 128)
    packed, state = bnb_functional.quantize_4bit(
        source,
        compress_statistics=True,
        quant_type="nf4",
    )
    torch.cuda.synchronize()
    if str(packed.dtype).removeprefix("torch.") != "uint8":
        raise RuntimeError("dynamic NF4 call did not produce uint8 packed weights")
    if getattr(state, "quant_type", None) != "nf4" or not bool(
        getattr(state, "nested", False)
    ):
        raise RuntimeError("dynamic NF4 call did not produce nested quant state")
    config_record = {
        "bnb_4bit_compute_dtype": str(config.bnb_4bit_compute_dtype).removeprefix(
            "torch."
        ),
        "bnb_4bit_quant_storage": str(config.bnb_4bit_quant_storage).removeprefix(
            "torch."
        ),
        "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": config.bnb_4bit_use_double_quant,
        "load_in_4bit": config.load_in_4bit,
    }
    expected_config = {
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "load_in_4bit": True,
    }
    if config_record != expected_config:
        raise RuntimeError(f"HF BitsAndBytesConfig differs: {config_record!r}")
    return {
        "bitsandbytes_loader_sha256": loader_hash,
        "bitsandbytes_version": importlib.metadata.version("bitsandbytes"),
        "dynamic_quant_call": {
            "compress_statistics": True,
            "input_dtype": "bfloat16",
            "nested_state": True,
            "packed_dtype": "uint8",
            "quant_type": "nf4",
        },
        "hf_generator_config": config_record,
    }


class _NvidiaMemorySampler:
    """Poll device-zero memory so model subprocess allocations are observed."""

    def __init__(self) -> None:
        self.total_bytes = 0
        self.peak_used_bytes = 0
        self.observation_count = 0
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> "_NvidiaMemorySampler":
        self._observe_once()
        self._thread.start()
        return self

    def __exit__(self, *unused: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("NVIDIA memory sampler did not stop")
        if self._error is not None:
            raise RuntimeError("NVIDIA memory sampler failed") from self._error
        self._observe_once()

    def _sample(self) -> None:
        try:
            while not self._stop.wait(0.10):
                self._observe_once()
        except BaseException as exc:
            with self._lock:
                self._error = exc
            self._stop.set()

    def _observe_once(self) -> None:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        if len(rows) != 1:
            raise RuntimeError("memory sampler requires exactly one GPU row")
        total_mib, used_mib = (int(part.strip()) for part in rows[0].split(","))
        total = total_mib * 1024**2
        used = used_mib * 1024**2
        with self._lock:
            if self.total_bytes not in (0, total):
                raise RuntimeError(
                    "reported GPU total memory changed during the sentinel"
                )
            self.total_bytes = total
            self.peak_used_bytes = max(self.peak_used_bytes, used)
            self.observation_count += 1


def _gmu_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
) -> dict[str, Any]:
    requirements = _mapping(planned_job.get("requirements"), "requirements")
    gmu = _finite_float(requirements.get("gpu_memory_utilization"), "gmu")
    measured = _model_capacity_measurements(
        plan_record=plan_record,
        planned_job=planned_job,
        input_bundle=input_bundle,
        work_dir=work_dir,
        gpu_memory_utilization=gmu,
        input_context_tokens=GPU_QUALIFICATION_INPUT_CONTEXT_TOKENS,
        max_model_len=GPU_QUALIFICATION_MAX_MODEL_LEN,
    )
    measured["candidate_qualified"] = (
        measured["kv_cache_capacity_tokens"]
        >= GPU_QUALIFICATION_REQUIRED_KV_CAPACITY_TOKENS
        and measured["observed_peak_headroom_bytes"]
        >= GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES
    )
    return measured


def _a10g_capacity_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
) -> dict[str, Any]:
    measured = _model_capacity_measurements(
        plan_record=plan_record,
        planned_job=planned_job,
        input_bundle=input_bundle,
        work_dir=work_dir,
        gpu_memory_utilization=0.90,
        input_context_tokens=GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS,
        max_model_len=GPU_QUALIFICATION_A10G_MAX_MODEL_LEN,
    )
    measured["capacity_qualified"] = (
        measured["kv_cache_capacity_tokens"]
        >= GPU_QUALIFICATION_A10G_REQUIRED_KV_CAPACITY_TOKENS
        and measured["observed_peak_headroom_bytes"]
        >= GPU_QUALIFICATION_MIN_PEAK_HEADROOM_BYTES
    )
    return measured


def _model_capacity_measurements(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
    gpu_memory_utilization: float,
    input_context_tokens: int,
    max_model_len: int,
) -> dict[str, Any]:
    """Execute the exact cold-disk Vanilla connector serving envelope at c4."""

    torch = _torch()
    probe = _triton_e5m2_probe(work_dir)
    weight_attestation = _weight_quantizer_attestation()
    selected_records = [
        _selected_jsonl_record(path)
        for path in _bucket_dataset_paths(input_bundle, input_context_tokens)
    ]
    selected_records.sort(key=lambda item: str(item.get("example_id", "")))
    selected_path = work_dir / "capacity-inputs.jsonl"
    _write_jsonl(selected_path, selected_records)

    _configure_transformers_generator(pre_rope=True)
    from document_kv_cache.benchmark_handoffs import (
        generate_benchmark_handoff_bundles,
    )
    from document_kv_cache.benchmarks import DEFAULT_V1_PROMPT_TEMPLATE_VERSION
    from document_kv_cache.model_profiles import (
        QWEN3_4B_ROPE_ROTARY_DIM,
        QWEN3_4B_ROPE_THETA,
        layout_for_model,
    )
    from document_kv_cache.models import CacheGenerationMethod
    from document_kv_cache.transformers_generator import (
        build_pre_rope_transformers_kv_chunk_generator,
    )
    from document_kv_cache.vllm_smoke import release_handoff_generation_resources

    layout = layout_for_model(
        GPU_QUALIFICATION_MODEL_ID,
        dtype="fp8_e5m2",
        block_size=16,
        lora_id="base",
        pre_rope=True,
        rope_theta=QWEN3_4B_ROPE_THETA,
        rope_rotary_dim=QWEN3_4B_ROPE_ROTARY_DIM,
    )
    generator = build_pre_rope_transformers_kv_chunk_generator()
    generator.bind_layout(layout)
    handoff_dir = work_dir / "capacity-handoffs"
    bundle = generate_benchmark_handoff_bundles(
        selected_path,
        output_dir=handoff_dir,
        generator=generator,
        layout=layout,
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version=DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
        cache_method=CacheGenerationMethod.VANILLA_PREFILL,
        model_revision=GPU_QUALIFICATION_MODEL_REVISION,
        tokenizer_id=GPU_QUALIFICATION_MODEL_ID,
        tokenizer_revision=GPU_QUALIFICATION_MODEL_REVISION,
        generator_family=generator.generator_family,
        generator_version=generator.generator_version,
        segment_per_document=True,
        require_artifact_contract=True,
        manifest_json=handoff_dir / "manifest.json",
    )
    params_by_key = {
        (entry.dataset, entry.example_id): entry.kv_transfer_params()
        for entry in bundle.manifest.entries
    }
    capacity_requests = _capacity_vanilla_requests(
        selected_records,
        params_by_key=params_by_key,
        input_context_tokens=input_context_tokens,
    )
    del bundle, generator
    release_handoff_generation_resources()
    gc.collect()
    torch.cuda.empty_cache()
    _fsync_tree(handoff_dir)
    evicted_files = _evict_tree_from_page_cache(handoff_dir)

    from document_kv_cache.vllm_smoke import (
        SERVED_MODEL_NAME,
        VLLMSmokeBenchmarkConfig,
        terminate_process,
        wait_for_server,
    )

    hardware_id = _required_string(planned_job, "hardware_id")
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id=f"gpu-qualification-{hardware_id}-capacity",
        output_dir=work_dir / "capacity-server-output",
        local_root=work_dir / "capacity-server-local",
        model_id=GPU_QUALIFICATION_MODEL_ID,
        model_revision=GPU_QUALIFICATION_MODEL_REVISION,
        tokenizer_revision=GPU_QUALIFICATION_MODEL_REVISION,
        model_dtype="bfloat16",
        model_quantization="bitsandbytes",
        kv_cache_dtype="fp8_e5m2",
        attention_backend="TRITON_ATTN",
        max_tokens=GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
        force_max_tokens=True,
        max_model_len=max_model_len,
        max_num_seqs=GPU_QUALIFICATION_REQUEST_PARALLELISM,
        gpu_memory_utilization=gpu_memory_utilization,
        benchmark_repeats=1,
        request_parallelism=GPU_QUALIFICATION_REQUEST_PARALLELISM,
        hardware_target=hardware_id,
        package_install_spec=str(Path(sys.prefix)),
    )
    config.output_dir.mkdir(parents=True)
    config.local_dir.mkdir(parents=True)
    server: subprocess.Popen[str] | None = None
    active_request_observation_count = 0
    with _NvidiaMemorySampler() as memory:
        server = _start_qualification_vllm_server(config)
        try:
            wait_for_server(
                server,
                config.server_log_path,
                config,
                timeout_seconds=1_200,
            )
            capacity = _server_kv_cache_capacity_tokens(config.server_log_path)
            _require_server_triton_backend(config.server_log_path)
            observations_before_requests = memory.observation_count
            request_results = _run_capacity_requests_concurrently(
                endpoint=f"{config.server_base_url}/v1/completions",
                model=SERVED_MODEL_NAME,
                requests=capacity_requests,
            )
            active_request_observation_count = (
                memory.observation_count - observations_before_requests
            )
        finally:
            terminate_process(server)
    server = None
    gc.collect()
    torch.cuda.empty_cache()
    if memory.total_bytes <= 0 or memory.peak_used_bytes <= 0:
        raise RuntimeError("GPU memory sampler returned no measurements")
    if active_request_observation_count < 2:
        raise RuntimeError(
            "GPU memory sampler did not span the active c4 request interval"
        )
    headroom = memory.total_bytes - memory.peak_used_bytes
    if headroom <= 0:
        raise RuntimeError("capacity sentinel exhausted measured GPU memory")
    successful_loads = _successful_connector_loads(config.connector_telemetry_path)
    expected_request_ids = {
        str(request["kv_transfer_params"]["document_kv.request_id"])
        for request in capacity_requests
    }
    observed_by_request = {
        str(record.get("request_id")): record
        for record in successful_loads
        if str(record.get("request_id")) in expected_request_ids
    }
    if set(observed_by_request) != expected_request_ids:
        raise RuntimeError("capacity batch did not load all four Vanilla handoffs")
    layer_counts: list[int] = []
    for request_id in sorted(expected_request_ids):
        counts = observed_by_request[request_id].get("counts")
        layers = counts.get("layers_loaded") if isinstance(counts, Mapping) else None
        if layers != GPU_QUALIFICATION_MODEL_LAYER_COUNT:
            raise RuntimeError(
                f"capacity connector request {request_id!r} did not load all layers"
            )
        layer_counts.append(int(layers))
    return {
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "cold_disk_evicted_file_count": evicted_files,
        "connector_loaded_layer_counts": layer_counts,
        "connector_successful_load_count": len(observed_by_request),
        "fatal_error_count": 0,
        "forced_decode_tokens": GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
        "gpu_memory_utilization": gpu_memory_utilization,
        "input_tokens_per_request": input_context_tokens,
        "kv_cache_capacity_tokens": capacity,
        "max_model_len": max_model_len,
        "observed_peak_headroom_bytes": headroom,
        "observed_peak_used_memory_bytes": memory.peak_used_bytes,
        "active_request_memory_observation_count": (
            active_request_observation_count
        ),
        "observed_total_memory_bytes": memory.total_bytes,
        "oom_count": 0,
        "q8_pre_rope_handoffs": True,
        "request_parallelism": GPU_QUALIFICATION_REQUEST_PARALLELISM,
        "request_success_count": len(request_results),
        "selected_examples": sorted(
            (
                {
                    "dataset": str(record["dataset"]),
                    "example_id": str(record["example_id"]),
                }
                for record in selected_records
            ),
            key=lambda item: (item["dataset"], item["example_id"]),
        ),
        "triton_compile_count": int(probe["triton_compile_count"]),
        "triton_kernel_launch_count": int(probe["triton_kernel_launch_count"]),
        "trust_remote_code": False,
        "vanilla_handoff_injected": True,
        "weight_quantizer_attestation": weight_attestation,
    }


def _capacity_vanilla_requests(
    selected_records: Sequence[Mapping[str, Any]],
    *,
    params_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    input_context_tokens: int,
) -> list[dict[str, Any]]:
    from document_kv_cache._benchmark_datasets import _example_from_record
    from document_kv_cache.benchmarks import (
        DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM,
        build_cache_prefix_text,
        build_cache_suffix_text,
        build_prefill_prompt,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        GPU_QUALIFICATION_MODEL_ID,
        revision=GPU_QUALIFICATION_MODEL_REVISION,
        trust_remote_code=False,
    )
    result: list[dict[str, Any]] = []
    for record in selected_records:
        dataset = str(record["dataset"])
        example_id = str(record["example_id"])
        example = _example_from_record(
            record,
            default_dataset=dataset,
            record_index=1,
            require_dataset=True,
        )
        full_ids = tokenizer.encode(
            build_prefill_prompt(example), add_special_tokens=False
        )
        prefix_ids = tokenizer.encode(
            build_cache_prefix_text(example), add_special_tokens=False
        )
        suffix = build_cache_suffix_text(example)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        if [int(item) for item in prefix_ids + suffix_ids] != [
            int(item) for item in full_ids
        ]:
            raise RuntimeError("capacity Vanilla token composition differs")
        if len(full_ids) != input_context_tokens:
            raise RuntimeError(
                f"capacity input has {len(full_ids)} tokens, expected "
                f"{input_context_tokens}"
            )
        params = params_by_key[(dataset, example_id)]
        runtime_prefix = params.get(DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM)
        if not isinstance(runtime_prefix, str):
            raise RuntimeError("capacity handoff runtime prefix is missing")
        result.append(
            {
                "cache_salt": f"qualification-capacity:{dataset}:{example_id}",
                "kv_transfer_params": dict(params),
                "prompt": f"{runtime_prefix}{suffix}",
            }
        )
    if len(result) != GPU_QUALIFICATION_REQUEST_PARALLELISM:
        raise RuntimeError("input bundle lacks four capacity prompts")
    return result


def _run_capacity_requests_concurrently(
    *,
    endpoint: str,
    model: str,
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(requests) != GPU_QUALIFICATION_REQUEST_PARALLELISM:
        raise RuntimeError("capacity request batch must contain exactly four requests")
    barrier = threading.Barrier(GPU_QUALIFICATION_REQUEST_PARALLELISM)

    def execute(request: Mapping[str, Any]) -> dict[str, Any]:
        barrier.wait(timeout=30)
        params = _mapping(
            request.get("kv_transfer_params"), "capacity kv_transfer_params"
        )
        return _completion_request(
            endpoint=endpoint,
            model=model,
            prompt=_required_string(request, "prompt"),
            kv_transfer_params=params,
            cache_salt=_required_string(request, "cache_salt"),
            max_tokens=GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
        )

    with ThreadPoolExecutor(
        max_workers=GPU_QUALIFICATION_REQUEST_PARALLELISM,
        thread_name_prefix="cachet-capacity-c4",
    ) as executor:
        return list(executor.map(execute, requests))


def _server_kv_cache_capacity_tokens(log_path: Path) -> int:
    if not log_path.is_file() or log_path.is_symlink():
        raise RuntimeError("vLLM server log is unavailable for capacity attestation")
    matches = re.findall(
        r"GPU KV cache size:\s*([0-9][0-9,]*)\s+tokens",
        log_path.read_text(encoding="utf-8", errors="replace"),
    )
    capacities = {int(value.replace(",", "")) for value in matches}
    if len(capacities) != 1 or next(iter(capacities)) <= 0:
        raise RuntimeError(
            f"vLLM server log has no unique positive KV capacity: {sorted(capacities)!r}"
        )
    return next(iter(capacities))


def _require_server_triton_backend(log_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"Using\s+[^\n]*TRITON_ATTN\s+backend\.", text) is None:
        raise RuntimeError("vLLM server log did not attest the forced TRITON_ATTN backend")


def _evict_tree_from_page_cache(root: Path) -> int:
    advise = getattr(os, "posix_fadvise", None)
    dont_need = getattr(os, "POSIX_FADV_DONTNEED", None)
    if not callable(advise) or type(dont_need) is not int:
        raise RuntimeError("cold-disk qualification requires POSIX_FADV_DONTNEED")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise RuntimeError("capacity handoff tree contains no files to evict")
    evicted = 0
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"capacity handoff contains a symlink: {path}")
        descriptor = os.open(path, os.O_RDONLY)
        try:
            advise(descriptor, 0, 0, dont_need)
        finally:
            os.close(descriptor)
        evicted += 1
    return evicted


def _kv_cache_capacity_tokens(llm: Any) -> int:
    candidates = [
        getattr(getattr(llm, "llm_engine", None), "vllm_config", None),
        getattr(getattr(llm, "llm_engine", None), "cache_config", None),
    ]
    for candidate in candidates:
        cache = getattr(candidate, "cache_config", candidate)
        blocks = getattr(cache, "num_gpu_blocks", None)
        block_size = getattr(cache, "block_size", None)
        if type(blocks) is int and blocks > 0 and type(block_size) is int and block_size > 0:
            return blocks * block_size
    raise RuntimeError("vLLM did not expose measured GPU KV-block capacity")


def _observed_attention_backends(llm: Any) -> set[str]:
    records = llm.apply_model(_model_attention_backend_names)
    names = {
        str(name)
        for record in records
        for name in (record if isinstance(record, Sequence) else ())
    }
    normalized: set[str] = set()
    for name in names:
        upper = name.upper()
        if "TRITON" in upper:
            normalized.add("TRITON_ATTN")
        elif "FLASHINFER" in upper:
            normalized.add("FLASHINFER")
        elif "FLASH_ATTN" in upper or "FLASHATTN" in upper:
            normalized.add("FLASH_ATTN")
        else:
            normalized.add(name)
    return normalized


def _model_attention_backend_names(model: Any) -> list[str]:
    names: set[str] = set()
    for module in model.modules():
        cls = type(module)
        qualified = f"{cls.__module__}.{cls.__name__}"
        upper = qualified.upper()
        if "ATTENTION" in upper or "TRITON_ATTN" in upper or "FLASHINFER" in upper:
            names.add(qualified)
        impl = getattr(module, "impl", None)
        if impl is not None:
            impl_cls = type(impl)
            names.add(f"{impl_cls.__module__}.{impl_cls.__name__}")
    return sorted(names)


def _shutdown_llm(llm: Any | None) -> None:
    if llm is None:
        return
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if not callable(shutdown):
        shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        raise RuntimeError("vLLM engine does not expose a shutdown method")
    shutdown()


def _auto_backend_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
) -> dict[str, Any]:
    del plan_record, planned_job, input_bundle, work_dir
    if os.environ.get("VLLM_ATTENTION_BACKEND"):
        raise RuntimeError("auto-backend diagnostic inherited a forced backend")
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    llm: Any = LLM(
        model=GPU_QUALIFICATION_MODEL_ID,
        revision=GPU_QUALIFICATION_MODEL_REVISION,
        tokenizer_revision=GPU_QUALIFICATION_MODEL_REVISION,
        dtype="bfloat16",
        quantization="bitsandbytes",
        kv_cache_dtype="fp8_e5m2",
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.70,
        trust_remote_code=False,
        seed=17,
        enforce_eager=True,
    )
    try:
        outputs = llm.generate(
            ["Return the single word cachet."],
            SamplingParams(temperature=0.0, seed=17, max_tokens=1),
            use_tqdm=False,
        )
        if len(outputs) != 1:
            raise RuntimeError("auto backend diagnostic request failed")
        backends = sorted(_observed_attention_backends(llm))
        if not backends:
            raise RuntimeError("auto backend diagnostic observed no attention backend")
        observed = "+".join(backends)
    finally:
        _shutdown_llm(llm)
    return {
        "backend_selection_mode": "auto",
        "observed_backend": observed,
        "publication_backend_changed": False,
        "trust_remote_code": False,
    }


def _throughput_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
) -> dict[str, Any]:
    del plan_record
    torch = _torch()
    probe = _triton_e5m2_probe(work_dir)
    _configure_transformers_generator(pre_rope=True)
    from document_kv_cache._benchmark_datasets import _example_from_record
    from document_kv_cache.benchmark_handoffs import (
        generate_benchmark_handoff_bundles,
    )
    from document_kv_cache.benchmarks import (
        DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
        benchmark_cache_prefix_segments,
    )
    from document_kv_cache.model_profiles import (
        QWEN3_4B_ROPE_ROTARY_DIM,
        QWEN3_4B_ROPE_THETA,
        layout_for_model,
    )
    from document_kv_cache.models import CacheGenerationMethod
    from document_kv_cache.storage import local_path
    from document_kv_cache.transformers_generator import (
        build_pre_rope_transformers_kv_chunk_generator,
    )

    layout = layout_for_model(
        GPU_QUALIFICATION_MODEL_ID,
        dtype="fp8_e5m2",
        block_size=16,
        lora_id="base",
        pre_rope=True,
        rope_theta=QWEN3_4B_ROPE_THETA,
        rope_rotary_dim=QWEN3_4B_ROPE_ROTARY_DIM,
    )
    generator = build_pre_rope_transformers_kv_chunk_generator()
    generator.bind_layout(layout)
    weight_attestation = _weight_quantizer_attestation()
    bucket_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    total_tokens = 0
    total_seconds = 0.0
    writes_dir = work_dir / "throughput-writes"
    writes_dir.mkdir()
    for length in GPU_QUALIFICATION_THROUGHPUT_BUCKETS:
        paths = _bucket_dataset_paths(input_bundle, length)
        if len(paths) != GPU_QUALIFICATION_THROUGHPUT_SAMPLES_PER_BUCKET:
            raise RuntimeError(f"throughput bucket {length} lacks four samples")
        start = time.perf_counter()
        completed_writes = 0
        bucket_prefix_tokens = 0
        for sample_index, path in enumerate(paths):
            record = _selected_jsonl_record(path)
            dataset = str(record["dataset"])
            example_id = str(record["example_id"])
            example = _example_from_record(
                record,
                default_dataset=dataset,
                record_index=1,
                require_dataset=True,
            )
            segments = benchmark_cache_prefix_segments(example)
            if len(segments) != length // 2_048:
                raise RuntimeError(
                    f"{dataset}:{example_id} has {len(segments)} segments; "
                    f"expected {length // 2_048} on the 2k document grid"
                )
            segment_contracts: list[dict[str, Any]] = []
            prefix_token_ids: list[int] = []
            for segment_index, (_chunk_id, text) in enumerate(segments):
                token_ids = _generator_token_ids(generator, text)
                prefix_token_ids.extend(token_ids)
                segment_contracts.append(
                    {
                        "index": segment_index,
                        "token_count": len(token_ids),
                        "token_ids_sha256": _integer_sequence_sha256(token_ids),
                    }
                )
            sample_input = writes_dir / f"{length}-{sample_index}.jsonl"
            _write_jsonl(sample_input, (record,))
            sample_dir = writes_dir / f"{length}-{sample_index}-bundle"
            bundle = generate_benchmark_handoff_bundles(
                sample_input,
                output_dir=sample_dir,
                generator=generator,
                layout=layout,
                model_id=layout.model_id,
                lora_id=layout.lora_id,
                prompt_template_version=DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
                cache_method=CacheGenerationMethod.VANILLA_PREFILL,
                model_revision=GPU_QUALIFICATION_MODEL_REVISION,
                tokenizer_id=GPU_QUALIFICATION_MODEL_ID,
                tokenizer_revision=GPU_QUALIFICATION_MODEL_REVISION,
                generator_family=generator.generator_family,
                generator_version=generator.generator_version,
                segment_per_document=True,
                require_artifact_contract=True,
                manifest_json=sample_dir / "manifest.json",
            )
            if len(bundle.payload_uris) != 1:
                raise RuntimeError("segmented pilot must produce one raw shard")
            if len(bundle.cache_refs) != len(segments):
                raise RuntimeError("generated refs do not match document segments")
            if [ref.token_count for ref in bundle.cache_refs] != [
                segment["token_count"] for segment in segment_contracts
            ]:
                raise RuntimeError("generated token contracts differ from inputs")
            raw_artifact = Path(local_path(bundle.payload_uris[0]))
            _fsync_tree(sample_dir)
            sample_records.append(
                {
                    "cache_prefix_token_count": len(prefix_token_ids),
                    "cache_prefix_token_ids_sha256": (
                        _integer_sequence_sha256(prefix_token_ids)
                    ),
                    "dataset": dataset,
                    "example_id": example_id,
                    "input_tokens_target": length,
                    "raw_artifact_bytes": raw_artifact.stat().st_size,
                    "raw_artifact_sha256": _file_sha256(raw_artifact),
                    "segment_count": len(segment_contracts),
                    "segments": segment_contracts,
                }
            )
            bucket_prefix_tokens += len(prefix_token_ids)
            completed_writes += 1
            del bundle
            gc.collect()
            torch.cuda.empty_cache()
        wall_seconds = time.perf_counter() - start
        prefix_tokens = bucket_prefix_tokens
        rate = prefix_tokens / wall_seconds
        if (
            _required_string(planned_job, "hardware_id")
            == "aws-g6e-l40s"
            and rate < GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND
        ):
            raise RuntimeError(
                f"throughput bucket {length} measured {rate:.3f} tokens/s, "
                f"below {GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND:.3f}"
            )
        bucket_records.append(
            {
                "durable_write_completed_count": completed_writes,
                "length_bucket_tokens": length,
                "prefix_tokens": prefix_tokens,
                "sample_count": len(paths),
                "tokens_per_second": rate,
                "wall_seconds": wall_seconds,
            }
        )
        total_tokens += prefix_tokens
        total_seconds += wall_seconds
    aggregate_rate = total_tokens / total_seconds
    if (
        _required_string(planned_job, "hardware_id") == "aws-g6e-l40s"
        and aggregate_rate < GPU_QUALIFICATION_MIN_PREFIX_TOKENS_PER_SECOND
    ):
        raise RuntimeError("aggregate generation throughput is below threshold")
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "aggregate_prefix_tokens": total_tokens,
        "aggregate_tokens_per_second": aggregate_rate,
        "aggregate_wall_seconds": total_seconds,
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "buckets": bucket_records,
        "clock_scope": "prefix_generation_through_durable_kv_write",
        "failed_write_count": 0,
        "generator_device_map": "auto",
        "samples": sorted(
            sample_records,
            key=lambda item: (
                int(item["input_tokens_target"]),
                str(item["dataset"]),
                str(item["example_id"]),
            ),
        ),
        "triton_compile_count": int(probe["triton_compile_count"]),
        "triton_kernel_launch_count": int(probe["triton_kernel_launch_count"]),
        "trust_remote_code": False,
        "weight_quantizer_attestation": weight_attestation,
        "writes_included": True,
    }


def _configure_transformers_generator(*, pre_rope: bool) -> None:
    from document_kv_cache.transformers_generator import (
        CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV,
        CACHET_TRANSFORMERS_DEVICE_ENV,
        CACHET_TRANSFORMERS_DEVICE_MAP_ENV,
        CACHET_TRANSFORMERS_MODEL_ID_ENV,
        CACHET_TRANSFORMERS_MODEL_REVISION_ENV,
        CACHET_TRANSFORMERS_PRE_ROPE_ENV,
        CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
        CACHET_TRANSFORMERS_QUANTIZATION_ENV,
        CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
        CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
        CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
        CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
    )

    values = {
        CACHET_TRANSFORMERS_MODEL_ID_ENV: GPU_QUALIFICATION_MODEL_ID,
        CACHET_TRANSFORMERS_MODEL_REVISION_ENV: GPU_QUALIFICATION_MODEL_REVISION,
        CACHET_TRANSFORMERS_TOKENIZER_ID_ENV: GPU_QUALIFICATION_MODEL_ID,
        CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV: (
            GPU_QUALIFICATION_MODEL_REVISION
        ),
        CACHET_TRANSFORMERS_DEVICE_ENV: "cuda",
        CACHET_TRANSFORMERS_DEVICE_MAP_ENV: "auto",
        CACHET_TRANSFORMERS_TORCH_DTYPE_ENV: "bfloat16",
        CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV: "0",
        CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV: "0",
        CACHET_TRANSFORMERS_QUANTIZATION_ENV: "bitsandbytes-4bit",
        CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV: json.dumps(
            {
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_storage": "uint8",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "load_in_4bit": True,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        CACHET_TRANSFORMERS_PRE_ROPE_ENV: "1" if pre_rope else "0",
    }
    for name, value in values.items():
        existing = os.environ.get(name)
        if existing not in (None, value):
            raise RuntimeError(
                f"generator environment {name} conflicts with qualification pins"
            )
        os.environ[name] = value


def _bucket_prompt_texts(input_bundle: Path, length: int) -> list[str]:
    from document_kv_cache._benchmark_datasets import _example_from_record
    from document_kv_cache.benchmarks import build_prefill_prompt

    prompts: list[str] = []
    for path in _bucket_dataset_paths(input_bundle, length):
        record = _selected_jsonl_record(path)
        example = _example_from_record(
            record,
            default_dataset=str(record["dataset"]),
            record_index=1,
            require_dataset=True,
        )
        prompts.append(build_prefill_prompt(example))
    return prompts


def _generator_token_ids(generator: Any, text: str) -> list[int]:
    tokenizer = getattr(generator, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        value = encode(text, add_special_tokens=False)
    elif callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        value = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    else:
        value = None
    if hasattr(value, "tolist"):
        value = value.tolist()
    while (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 1
        and isinstance(value[0], Sequence)
    ):
        value = value[0]
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise RuntimeError("generator tokenizer did not return integer token IDs")
    return [int(item) for item in value]


def _fsync_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"durable output root is not a directory: {root}")
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"durable output contains a symlink: {path}")
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
        elif path.is_dir():
            descriptor = os.open(path, os.O_RDONLY)
        else:
            raise RuntimeError(f"durable output contains a special file: {path}")
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _matched_token_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Run baseline and Vanilla through the real HTTP connector boundary."""

    torch = _torch()
    probe = _triton_e5m2_probe(work_dir)
    selected_path = work_dir / "matched-inputs.jsonl"
    selected_records = [
        _selected_jsonl_record(path)
        for path in _bucket_dataset_paths(input_bundle, 8_192)
    ]
    selected_records.sort(key=lambda item: str(item.get("example_id", "")))
    _write_jsonl(selected_path, selected_records)

    _configure_transformers_generator(pre_rope=True)
    from document_kv_cache.benchmark_handoffs import (
        generate_benchmark_handoff_bundles,
    )
    from document_kv_cache.benchmarks import DEFAULT_V1_PROMPT_TEMPLATE_VERSION
    from document_kv_cache.model_profiles import (
        QWEN3_4B_ROPE_ROTARY_DIM,
        QWEN3_4B_ROPE_THETA,
        layout_for_model,
    )
    from document_kv_cache.models import CacheGenerationMethod
    from document_kv_cache.transformers_generator import (
        build_pre_rope_transformers_kv_chunk_generator,
    )
    from document_kv_cache.vllm_smoke import release_handoff_generation_resources

    layout = layout_for_model(
        GPU_QUALIFICATION_MODEL_ID,
        dtype="fp8_e5m2",
        block_size=16,
        lora_id="base",
        pre_rope=True,
        rope_theta=QWEN3_4B_ROPE_THETA,
        rope_rotary_dim=QWEN3_4B_ROPE_ROTARY_DIM,
    )
    generator = build_pre_rope_transformers_kv_chunk_generator()
    generator.bind_layout(layout)
    handoff_dir = work_dir / "matched-handoffs"
    bundle = generate_benchmark_handoff_bundles(
        selected_path,
        output_dir=handoff_dir,
        generator=generator,
        layout=layout,
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version=DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
        cache_method=CacheGenerationMethod.VANILLA_PREFILL,
        model_revision=GPU_QUALIFICATION_MODEL_REVISION,
        tokenizer_id=GPU_QUALIFICATION_MODEL_ID,
        tokenizer_revision=GPU_QUALIFICATION_MODEL_REVISION,
        generator_family=generator.generator_family,
        generator_version=generator.generator_version,
        segment_per_document=True,
        require_artifact_contract=True,
        manifest_json=handoff_dir / "manifest.json",
    )
    params_by_key = {
        (entry.dataset, entry.example_id): entry.kv_transfer_params()
        for entry in bundle.manifest.entries
    }
    del bundle, generator
    release_handoff_generation_resources()
    gc.collect()
    torch.cuda.empty_cache()

    from document_kv_cache.vllm_smoke import (
        SERVED_MODEL_NAME,
        VLLMSmokeBenchmarkConfig,
        terminate_process,
        wait_for_server,
    )

    hardware_id = _required_string(planned_job, "hardware_id")
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id=f"gpu-qualification-{hardware_id}-matched-token",
        output_dir=work_dir / "matched-server-output",
        local_root=work_dir / "matched-server-local",
        model_id=GPU_QUALIFICATION_MODEL_ID,
        model_revision=GPU_QUALIFICATION_MODEL_REVISION,
        tokenizer_revision=GPU_QUALIFICATION_MODEL_REVISION,
        model_dtype="bfloat16",
        model_quantization="bitsandbytes",
        kv_cache_dtype="fp8_e5m2",
        attention_backend="TRITON_ATTN",
        max_tokens=16,
        force_max_tokens=True,
        max_model_len=8_256,
        max_num_seqs=1,
        gpu_memory_utilization=0.75 if hardware_id == "aws-g6-l4" else 0.90,
        benchmark_repeats=2,
        request_parallelism=1,
        hardware_target=hardware_id,
        package_install_spec=str(Path(sys.prefix)),
    )
    config.output_dir.mkdir(parents=True)
    config.local_dir.mkdir(parents=True)
    server = _start_qualification_vllm_server(config)
    try:
        wait_for_server(
            server,
            config.server_log_path,
            config,
            timeout_seconds=1_200,
        )
        results = _run_matched_http_requests(
            config=config,
            selected_records=selected_records,
            params_by_key=params_by_key,
            input_bundle_sha256=_plan_artifact_pin(
                plan_record, "input_bundle_sha256"
            ),
            served_model_name=SERVED_MODEL_NAME,
        )
    finally:
        terminate_process(server)
    successful_loads = _successful_connector_loads(config.connector_telemetry_path)
    expected_request_ids = {
        str(params["document_kv.request_id"]) for params in params_by_key.values()
    }
    observed_request_ids = {
        str(record.get("request_id")) for record in successful_loads
    }
    if not expected_request_ids.issubset(observed_request_ids):
        raise RuntimeError(
            "Vanilla HTTP requests did not produce successful connector loads for "
            f"{sorted(expected_request_ids - observed_request_ids)!r}"
        )
    for record in successful_loads:
        counts = record.get("counts")
        if isinstance(counts, Mapping) and record.get("request_id") in expected_request_ids:
            if counts.get("layers_loaded") != GPU_QUALIFICATION_MODEL_LAYER_COUNT:
                raise RuntimeError("connector load did not inject all model layers")
    return {
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "baseline_handoff_absent": True,
        "cache_phases": ["cold", "warm"],
        "execution_mode": "real_end_to_end_requests",
        "examples": results,
        "request_parallelism": 1,
        "triton_compile_count": int(probe["triton_compile_count"]),
        "triton_kernel_launch_count": int(probe["triton_kernel_launch_count"]),
        "vanilla_handoff_injected": True,
        "trust_remote_code": False,
    }


def _start_qualification_vllm_server(config: Any) -> subprocess.Popen[str]:
    """Start via the smoke CLI builder with remote code explicitly disabled."""

    from document_kv_cache.vllm_smoke import build_vllm_server_args, server_env

    argv = build_vllm_server_args(config, Path(sys.executable))
    if "--trust-remote-code" in argv:
        raise RuntimeError("vLLM server args unexpectedly enabled trust_remote_code")
    config.server_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = config.server_log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_env(config),
        )
    finally:
        log_handle.close()


def _run_matched_http_requests(
    *,
    config: Any,
    selected_records: Sequence[Mapping[str, Any]],
    params_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    input_bundle_sha256: str,
    served_model_name: str,
) -> list[dict[str, Any]]:
    from document_kv_cache._benchmark_datasets import _example_from_record
    from document_kv_cache.benchmarks import (
        DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM,
        build_cache_prefix_text,
        build_cache_suffix_text,
        build_prefill_prompt,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        GPU_QUALIFICATION_MODEL_ID,
        revision=GPU_QUALIFICATION_MODEL_REVISION,
        trust_remote_code=False,
    )
    endpoint = f"{config.server_base_url}/v1/completions"
    results: list[dict[str, Any]] = []
    for record in selected_records:
        dataset = str(record["dataset"])
        example_id = str(record["example_id"])
        example = _example_from_record(
            record,
            default_dataset=dataset,
            record_index=1,
            require_dataset=True,
        )
        full_prompt = build_prefill_prompt(example)
        cache_prefix = build_cache_prefix_text(example)
        cache_suffix = build_cache_suffix_text(example)
        full_ids = [
            int(item)
            for item in tokenizer.encode(full_prompt, add_special_tokens=False)
        ]
        prefix_ids = [
            int(item)
            for item in tokenizer.encode(cache_prefix, add_special_tokens=False)
        ]
        suffix_ids = [
            int(item)
            for item in tokenizer.encode(cache_suffix, add_special_tokens=False)
        ]
        if prefix_ids + suffix_ids != full_ids:
            raise RuntimeError(
                f"Vanilla token composition differs for {dataset}:{example_id}"
            )
        if len(full_ids) != 8_192:
            raise RuntimeError("matched-token input is not the exact 8k bucket")
        params = params_by_key[(dataset, example_id)]
        runtime_prefix = params.get(DOCUMENT_KV_RUNTIME_PREFIX_TEXT_PARAM, "")
        if not isinstance(runtime_prefix, str):
            raise RuntimeError("handoff runtime prefix text is not a string")
        vanilla_prompt = f"{runtime_prefix}{cache_suffix}"
        position_hash = _integer_sequence_sha256(
            range(len(full_ids), len(full_ids) + 16)
        )
        arms: list[dict[str, Any]] = []
        for arm_id, prompt, handoff_params in (
            ("baseline_prefill", full_prompt, None),
            ("vanilla_prefill", vanilla_prompt, params),
        ):
            repeats = [
                _completion_request(
                    endpoint=endpoint,
                    model=served_model_name,
                    prompt=prompt,
                    kv_transfer_params=handoff_params,
                    cache_salt=(
                        f"qualification:{dataset}:{example_id}:{arm_id}:repeat-{repeat}"
                    ),
                )
                for repeat in range(2)
            ]
            token_hashes = [
                _integer_sequence_sha256(item["token_ids"]) for item in repeats
            ]
            if len(set(token_hashes)) != 1:
                raise RuntimeError(
                    f"non-deterministic token IDs for {dataset}:{example_id}:{arm_id}"
                )
            logprob_vectors = [item["token_logprobs"] for item in repeats]
            if len(logprob_vectors[0]) != len(logprob_vectors[1]):
                raise RuntimeError("determinism logprob lengths differ")
            drift = max(
                abs(float(left) - float(right))
                for left, right in zip(
                    logprob_vectors[0], logprob_vectors[1], strict=True
                )
            )
            if drift > GPU_QUALIFICATION_MAX_LOGIT_DRIFT:
                raise RuntimeError(
                    f"logit drift {drift} exceeds the qualification bound"
                )
            arms.append(
                {
                    "arm_id": arm_id,
                    "finite_logits": all(
                        math.isfinite(float(value))
                        for vector in logprob_vectors
                        for value in vector
                    ),
                    "logit_probe_position_ids_sha256": position_hash,
                    "max_abs_logit_drift": drift,
                    "output_token_count": len(repeats[0]["token_ids"]),
                    "output_token_ids_repeat_sha256": token_hashes,
                    "repeat_count": 2,
                }
            )
        token_hash = _integer_sequence_sha256(full_ids)
        results.append(
            {
                "arms": arms,
                "baseline_full_prompt_token_ids_sha256": token_hash,
                "example_id": example_id,
                "full_prompt_token_count": len(full_ids),
                "input_bundle_sha256": input_bundle_sha256,
                "vanilla_prefix_token_count": len(prefix_ids),
                "vanilla_reconstructed_full_prompt_token_ids_sha256": token_hash,
                "vanilla_suffix_token_count": len(suffix_ids),
            }
        )
    results.sort(key=lambda item: str(item["example_id"]))
    return results


def _completion_request(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    kv_transfer_params: Mapping[str, Any] | None,
    cache_salt: str,
    max_tokens: int = 16,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 17,
        "ignore_eos": True,
        "logprobs": 1,
        "return_token_ids": True,
        "cache_salt": cache_salt,
    }
    if kv_transfer_params is not None:
        body["kv_transfer_params"] = dict(kv_transfer_params)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        record = json.loads(response.read().decode("utf-8"))
    choices = record.get("choices") if isinstance(record, Mapping) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError(f"completion response has invalid choices: {record!r}")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("completion choice is not an object")
    token_ids = choice.get("token_ids")
    logprobs = choice.get("logprobs")
    token_logprobs = (
        logprobs.get("token_logprobs") if isinstance(logprobs, Mapping) else None
    )
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != max_tokens
        or any(type(item) is not int for item in token_ids)
    ):
        raise RuntimeError(
            f"completion did not return exactly {max_tokens} token IDs"
        )
    if (
        not isinstance(token_logprobs, list)
        or len(token_logprobs) != len(token_ids)
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in token_logprobs
        )
    ):
        raise RuntimeError("completion logprobs are missing or non-finite")
    return {
        "token_ids": token_ids,
        "token_logprobs": [float(item) for item in token_logprobs],
    }


def _successful_connector_loads(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("connector telemetry was not written")
    records: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if (
            isinstance(value, Mapping)
            and value.get("record_type")
            == "document_kv.vllm_native_provider_load.v1"
            and value.get("event") == "load_request"
            and value.get("success") is True
        ):
            records.append(value)
    if not records:
        raise RuntimeError("connector telemetry contains no successful loads")
    return records


def _bucket_dataset_paths(input_bundle: Path, length: int) -> list[Path]:
    directory = input_bundle / str(length)
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError(f"input bundle is missing the {length} bucket")
    paths = [directory / f"{dataset}.jsonl" for dataset in GPU_QUALIFICATION_INPUT_DATASETS]
    observed = {path.name for path in directory.glob("*.jsonl")}
    expected = {path.name for path in paths}
    if observed != expected or any(not path.is_file() or path.is_symlink() for path in paths):
        raise RuntimeError(
            f"input bucket {length} must contain exactly {sorted(expected)!r}"
        )
    return paths


def _selected_jsonl_record(path: Path) -> dict[str, Any]:
    """Select the lexicographically smallest example ID from a frozen shard."""

    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"prepared dataset is not a regular file: {path}")
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise RuntimeError(f"prepared dataset row is not an object: {path}")
        records.append(value)
    if len(records) != GPU_QUALIFICATION_INPUT_ROWS_PER_DATASET_BUCKET:
        raise RuntimeError(
            "qualification input must contain exactly "
            f"{GPU_QUALIFICATION_INPUT_ROWS_PER_DATASET_BUCKET} rows per "
            f"dataset/bucket: {path}"
        )
    identities: list[str] = []
    datasets: set[str] = set()
    for record in records:
        example_id = record.get("example_id")
        dataset = record.get("dataset")
        if not isinstance(example_id, str) or not example_id:
            raise RuntimeError(f"qualification row has no example_id: {path}")
        if not isinstance(dataset, str) or not dataset:
            raise RuntimeError(f"qualification row has no dataset: {path}")
        identities.append(example_id)
        datasets.add(dataset)
    if len(set(identities)) != len(identities) or len(datasets) != 1:
        raise RuntimeError(
            f"qualification shard has duplicate IDs or mixed datasets: {path}"
        )
    if datasets != {path.stem}:
        raise RuntimeError(
            f"qualification shard dataset does not match its filename: {path}"
        )
    selected_id = min(identities)
    return next(record for record in records if record["example_id"] == selected_id)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for record in records
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _plan_artifact_pin(plan_record: Mapping[str, Any], key: str) -> str:
    runtime = _mapping(plan_record.get("runtime_contract"), "runtime_contract")
    pins = _mapping(runtime.get("artifact_sha256"), "artifact_sha256")
    value = pins.get(key)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"plan artifact pin {key!r} is invalid")
    return value


def _integer_sequence_sha256(values: Sequence[int] | range) -> str:
    normalized = list(values)
    if any(type(value) is not int for value in normalized):
        raise TypeError("integer hash input contains a non-integer")
    return sha256(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_exists(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _torch() -> Any:
    import torch  # type: ignore[import-not-found]

    return torch


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{field_name} must be an object")
    return value


def _finite_float(value: Any, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def _finite_json_object(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{field_name} must be an object")
    return normalized


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one package-owned GPU qualification sentinel."
    )
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--job-json", required=True, type=Path)
    parser.add_argument("--input-bundle", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for path, label in (
        (args.plan_json, "plan"),
        (args.job_json, "job"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} JSON must be one regular file")
    if args.output_json.exists() or args.output_json.is_symlink():
        raise FileExistsError(f"measurement output already exists: {args.output_json}")
    plan = json.loads(args.plan_json.read_text(encoding="utf-8"))
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    if not isinstance(plan, Mapping) or not isinstance(job, Mapping):
        raise ValueError("plan and job JSON must contain objects")
    measurements = execute_planned_sentinel(
        plan_record=plan,
        planned_job=job,
        input_bundle=args.input_bundle,
        work_dir=args.work_dir,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.output_json,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o640,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_gpu_qualification_json(measurements) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        args.output_json.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover - GPU entry point.
    raise SystemExit(main())
