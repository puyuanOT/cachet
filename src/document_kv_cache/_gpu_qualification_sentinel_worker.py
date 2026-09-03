"""GPU-only sentinel worker for the closed vLLM 0.27.1 qualification plan.

This module is deliberately not imported by the Databricks payload renderer.
It runs inside the hash-locked Python 3.11/cu129 environment created by
``gpu_qualification_sentinels``.  Every dispatch target below is package owned;
there is no factory flag and no path that promotes caller-provided JSON to
qualification evidence.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Final, NoReturn

from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_A10G_INPUT_CONTEXT_TOKENS,
    GPU_QUALIFICATION_A10G_MAX_MODEL_LEN,
    GPU_QUALIFICATION_A10G_REQUIRED_KV_CAPACITY_TOKENS,
    GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256,
    GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
    GPU_QUALIFICATION_CONNECTOR_SOURCE_SHA256,
    GPU_QUALIFICATION_DETERMINISM_REPEATS,
    GPU_QUALIFICATION_MAX_E5M2_BF16_ERROR,
    GPU_QUALIFICATION_MAX_LOGIT_DRIFT,
    GPU_QUALIFICATION_MATCHED_EXAMPLES,
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
_NATIVE_SHARED_OBJECT_DISTRIBUTIONS: Final = (
    "bitsandbytes",
    "torch",
    "triton",
    "vllm",
)
_NATIVE_SHARED_OBJECT_NAME_RE: Final = re.compile(r".+\.so(?:\..+)?\Z")
_NATIVE_LDD_MAX_STREAM_BYTES: Final = 256 * 1024
_QUALIFICATION_HTTP_ERROR_MAX_BYTES: Final = 16 * 1024
_QUALIFICATION_COMPLETION_RESPONSE_MAX_BYTES: Final = 256 * 1024
_QUALIFICATION_SERVER_LOG_TAIL_MAX_BYTES: Final = 256 * 1024
_QUALIFICATION_FAILURE_DIAGNOSTIC_MAX_BYTES: Final = 4 * 1024
_QUALIFICATION_COMPLETION_PHASES: Final = frozenset({"capacity", "matched_token"})
_QUALIFICATION_COMPLETION_ARMS: Final = frozenset(
    {"baseline_prefill", "vanilla_prefill"}
)
_NATIVE_SELECTOR_ENVIRONMENT_NAMES: Final = (
    "BNB_CUDA_VERSION",
    "LLVM_PASS_PLUGIN_PATH",
    "TRITON_BACKENDS_IN_TREE",
    "TRITON_DEFAULT_BACKEND",
    "TRITON_PLUGIN_PATHS",
)
_SELECTED_BITSANDBYTES_NATIVE_LIBRARY_MEMBER: Final = (
    "bitsandbytes/libbitsandbytes_cuda129.so"
)
_PLATFORM_INAPPLICABLE_NATIVE_MISSING_SONAMES: Final = {
    (
        "bitsandbytes",
        "bitsandbytes/libbitsandbytes_cuda118.so",
    ): frozenset(
        {
            "libcublas.so.11",
            "libcublasLt.so.11",
            "libcudart.so.11.0",
            "libcusparse.so.11",
        }
    ),
    (
        "bitsandbytes",
        "bitsandbytes/libbitsandbytes_cuda130.so",
    ): frozenset(
        {
            "libcublas.so.13",
            "libcublasLt.so.13",
            "libcudart.so.13",
            "libnvJitLink.so.13",
        }
    ),
    (
        "bitsandbytes",
        "bitsandbytes/libbitsandbytes_rocm62.so",
    ): frozenset({"libhipblas.so.2", "libhipblaslt.so.0", "libhipsparse.so.1"}),
    (
        "bitsandbytes",
        "bitsandbytes/libbitsandbytes_rocm63.so",
    ): frozenset({"libhipblas.so.2", "libhipblaslt.so.0", "libhipsparse.so.1"}),
    (
        "bitsandbytes",
        "bitsandbytes/libbitsandbytes_rocm64.so",
    ): frozenset({"libhipblas.so.2", "libhipblaslt.so.0", "libhipsparse.so.1"}),
    (
        "bitsandbytes",
        "bitsandbytes/libbitsandbytes_rocm70.so",
    ): frozenset({"libhipblas.so.3", "libhipblaslt.so.1", "libhipsparse.so.4"}),
    (
        "bitsandbytes",
        "bitsandbytes/libbitsandbytes_rocm71.so",
    ): frozenset({"libhipblas.so.3", "libhipblaslt.so.1", "libhipsparse.so.4"}),
    (
        "bitsandbytes",
        "bitsandbytes/libbitsandbytes_rocm72.so",
    ): frozenset({"libhipblas.so.3", "libhipblaslt.so.1", "libhipsparse.so.4"}),
    (
        "bitsandbytes",
        "bitsandbytes/libbitsandbytes_xpu.so",
    ): frozenset(
        {"libimf.so", "libintlc.so.5", "libirng.so", "libsvml.so", "libsycl.so.8"}
    ),
    (
        "triton",
        "triton/plugins/libMLIRDialectPlugin.so",
    ): frozenset({"libtriton.so"}),
    (
        "triton",
        "triton/plugins/libMLIRDialectPlugin.so.23.0git",
    ): frozenset({"libtriton.so"}),
    (
        "triton",
        "triton/plugins/libTritonPluginsTestLib.so",
    ): frozenset({"libtriton.so"}),
}


class _QualificationCompletionFailure(RuntimeError):
    """Carry only bounded, non-sensitive completion failure evidence."""

    def __init__(
        self,
        *,
        request_id: str,
        context: Mapping[str, Any],
        request_diagnostic: Mapping[str, Any],
    ) -> None:
        super().__init__("qualification completion request failed")
        self.request_id = request_id
        self.context = dict(context)
        self.request_diagnostic = dict(request_diagnostic)


class _QualificationCompletionBatchFailure(RuntimeError):
    """Aggregate deterministic concurrent completion failures."""

    def __init__(self, failures: Sequence[_QualificationCompletionFailure]) -> None:
        super().__init__("qualification completion request batch failed")
        self.failures = tuple(failures)


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


def _reject_native_selector_environment() -> None:
    selected = sorted(
        name for name in _NATIVE_SELECTOR_ENVIRONMENT_NAMES if name in os.environ
    )
    if selected:
        raise RuntimeError(
            "GPU qualification forbids native-library selector environment names: "
            + ", ".join(selected)
        )


def execute_planned_sentinel(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    input_bundle: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Run exactly one plan-selected sentinel and return measured fields."""

    _reject_native_selector_environment()
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
    sequence_lengths = torch.tensor([token_count], device="cuda", dtype=torch.int32)
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
    scores = torch.einsum("qhd,thd->qht", query.float(), decoded_key) * softmax_scale
    reference = torch.einsum(
        "qht,thd->qhd", torch.softmax(scores, dim=-1), decoded_value
    )
    max_error = float((output.float() - reference).abs().max().item())
    if (
        not math.isfinite(max_error)
        or max_error > GPU_QUALIFICATION_MAX_E5M2_BF16_ERROR
    ):
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
            destination[..., :head_size].permute(0, 2, 1, 3).view(torch.float8_e5m2)
        )
        value_cache = (
            destination[..., head_size:].permute(0, 2, 1, 3).view(torch.float8_e5m2)
        )
        descale = torch.ones((1, heads), device="cuda", dtype=torch.float32)
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            max_seqlen_q=1,
            seqused_k=torch.tensor([token_count], device="cuda", dtype=torch.int32),
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
        scores = torch.einsum("qhd,thd->qht", query.float(), decoded_key) * (
            head_size**-0.5
        )
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
    runtime_lock_attestation = _runtime_lock_attestation_for_plan(plan_record)
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
    weight_attestation = _weight_quantizer_attestation()
    native_evidence = _native_shared_object_resolution()
    native_count = len(native_evidence)
    unresolved_count = sum(
        any(binding["resolved_path"] is None for binding in record["soname_bindings"])
        for record in native_evidence
    )
    unresolved_runtime_reachable_count = sum(
        record["resolution_scope"] == "runtime_reachable"
        and any(
            binding["resolved_path"] is None for binding in record["soname_bindings"]
        )
        for record in native_evidence
    )
    if unresolved_runtime_reachable_count != 0:
        raise RuntimeError("runtime-reachable native shared objects remain unresolved")
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
        "native_shared_object_evidence": list(native_evidence),
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
        "unresolved_runtime_reachable_native_shared_object_count": (
            unresolved_runtime_reachable_count
        ),
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


def _runtime_lock_attestation_for_plan(
    plan_record: Mapping[str, Any],
) -> dict[str, Any]:
    if plan_record.get("record_type") != ("cachet.vllm_0271_gpu_qualification_plan.v2"):
        return _runtime_lock_attestation()
    from document_kv_cache.gpu_qualification_v2 import (
        GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE,
        GPU_QUALIFICATION_V2_SCHEMA_VERSION,
        validate_gpu_qualification_v2_runtime_attestation,
    )

    if (
        plan_record.get("record_type") != GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE
        or type(plan_record.get("schema_version")) is not int
        or plan_record.get("schema_version") != GPU_QUALIFICATION_V2_SCHEMA_VERSION
    ):
        raise RuntimeError("full runtime-lock attestation has an open v2 plan schema")
    raw = os.environ.get(_RUNTIME_LOCK_ATTESTATION_ENV)
    if not raw:
        raise RuntimeError("full v2 runtime-lock attestation is unavailable")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("full v2 runtime-lock attestation is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("full v2 runtime-lock attestation must contain an object")
    validate_gpu_qualification_v2_runtime_attestation(value)
    return value


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
                tokens * GPU_QUALIFICATION_MODEL_LAYER_COUNT * 2 * heads * head_size,
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


def _pip_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for variable_name in tuple(environment):
        if variable_name.upper().startswith("PIP_"):
            environment.pop(variable_name)
    for variable_name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(variable_name, None)
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _pip_check_ok() -> bool:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=300,
        env=_pip_subprocess_environment(),
    )
    return completed.returncode == 0


def _native_shared_object_resolution(
    distribution_names: Sequence[str] = _NATIVE_SHARED_OBJECT_DISTRIBUTIONS,
) -> tuple[dict[str, Any], ...]:
    """Audit only shared objects owned by the requested distributions.

    ``Distribution.locate_file("")`` is commonly the shared ``site-packages``
    root.  Recursing from it therefore audits unrelated packages.  The wheel
    member inventory is the ownership authority; each candidate is located
    individually and its final target must remain within that same inventory.
    """

    names = _canonical_native_distribution_names(distribution_names)
    owned_objects: list[dict[str, Any]] = []
    observed_paths: dict[str, str] = {}
    for distribution_name in names:
        distribution = importlib.metadata.distribution(distribution_name)
        distribution_version = str(distribution.version)
        if (
            not distribution_version
            or distribution_version.strip() != distribution_version
            or any(ord(character) < 32 for character in distribution_version)
        ):
            raise RuntimeError(
                f"native distribution {distribution_name!r} has an invalid version"
            )
        raw_files = distribution.files
        if raw_files is None:
            raise RuntimeError(
                f"native distribution {distribution_name!r} has no owned file inventory"
            )
        members: dict[str, PurePosixPath] = {}
        for raw_member in raw_files:
            member = _native_shared_object_member(
                str(raw_member), distribution_name=distribution_name
            )
            if member is None:
                continue
            member_text = member.as_posix()
            if member_text in members:
                raise RuntimeError(
                    f"native distribution {distribution_name!r} repeats member "
                    f"{member_text!r}"
                )
            members[member_text] = member
        if not members:
            raise RuntimeError(
                f"native distribution {distribution_name!r} owns no shared objects"
            )

        root_path = _canonical_native_located_path(
            distribution.locate_file(""),
            label=f"native distribution {distribution_name!r} root",
        )
        _require_no_native_symlink_ancestors(
            root_path,
            label=f"native distribution {distribution_name!r} root",
            include_leaf=True,
        )
        try:
            resolved_root = root_path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"native distribution {distribution_name!r} root is unavailable"
            ) from exc
        if not resolved_root.is_dir():
            raise RuntimeError(
                f"native distribution {distribution_name!r} root is not a directory"
            )

        distribution_objects: list[dict[str, Any]] = []
        for member_text, member in sorted(members.items()):
            located_path = _canonical_native_located_path(
                distribution.locate_file(member),
                label=(
                    f"native distribution {distribution_name!r} member "
                    f"{member_text!r} located path"
                ),
            )
            expected_path = root_path.joinpath(*member.parts)
            if located_path != expected_path:
                raise RuntimeError(
                    f"native distribution {distribution_name!r} member "
                    f"{member_text!r} locates outside its canonical member path: "
                    f"{located_path}"
                )
            _require_no_native_symlink_ancestors(
                located_path,
                label=(
                    f"native distribution {distribution_name!r} member {member_text!r}"
                ),
                include_leaf=False,
            )
            try:
                member_status = located_path.lstat()
                resolved_path = located_path.resolve(strict=True)
                resolved_status = resolved_path.stat()
            except OSError as exc:
                raise RuntimeError(
                    f"native distribution {distribution_name!r} member "
                    f"{member_text!r} is unavailable at {located_path}"
                ) from exc
            is_symlink = stat.S_ISLNK(member_status.st_mode)
            if not is_symlink and not stat.S_ISREG(member_status.st_mode):
                raise RuntimeError(
                    f"native distribution {distribution_name!r} member "
                    f"{member_text!r} is not a regular file"
                )
            if not stat.S_ISREG(resolved_status.st_mode):
                raise RuntimeError(
                    f"native distribution {distribution_name!r} member "
                    f"{member_text!r} does not resolve to a regular file"
                )
            if not resolved_path.is_relative_to(resolved_root):
                raise RuntimeError(
                    f"native distribution {distribution_name!r} member "
                    f"{member_text!r} escapes its distribution root: {resolved_path}"
                )
            path_text = str(located_path)
            previous_owner = observed_paths.setdefault(path_text, distribution_name)
            if previous_owner != distribution_name:
                raise RuntimeError(
                    f"native shared object path {path_text!r} has multiple owners: "
                    f"{previous_owner!r}, {distribution_name!r}"
                )
            distribution_objects.append(
                {
                    "distribution": distribution_name,
                    "distribution_version": distribution_version,
                    "is_symlink": is_symlink,
                    "member": member_text,
                    "path": path_text,
                    "resolved_path": str(resolved_path),
                }
            )

        regular_targets = {
            item["resolved_path"]
            for item in distribution_objects
            if item["is_symlink"] is False
        }
        for item in distribution_objects:
            if (
                item["is_symlink"] is True
                and item["resolved_path"] not in regular_targets
            ):
                raise RuntimeError(
                    f"native distribution {distribution_name!r} symlink member "
                    f"{item['member']!r} targets an unowned object: "
                    f"{item['resolved_path']}"
                )
        owned_objects.extend(distribution_objects)

    if not owned_objects:
        raise RuntimeError("no owned native runtime shared objects were found")

    observed_owned_members = {
        (item["distribution"], item["member"]) for item in owned_objects
    }
    required_owned_members = {
        key for key in _PLATFORM_INAPPLICABLE_NATIVE_MISSING_SONAMES if key[0] in names
    }
    if "bitsandbytes" in names:
        required_owned_members.add(
            ("bitsandbytes", _SELECTED_BITSANDBYTES_NATIVE_LIBRARY_MEMBER)
        )
    missing_required_members = required_owned_members - observed_owned_members
    if missing_required_members:
        raise RuntimeError(
            "native distribution inventory lacks required reviewed members: "
            + ", ".join(
                f"{distribution}:{member}"
                for distribution, member in sorted(missing_required_members)
            )
        )

    ldd_environment = dict(os.environ)
    ldd_environment.update({"LANG": "C", "LC_ALL": "C"})
    evidence: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for owned in sorted(
        owned_objects,
        key=lambda item: (item["distribution"], item["member"], item["path"]),
    ):
        try:
            completed = subprocess.run(
                ["ldd", str(owned["path"])],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                timeout=30,
                env=ldd_environment,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise RuntimeError(
                "ldd could not audit owned native shared object "
                f"{owned['distribution']}:{owned['member']} at {owned['path']}"
            ) from exc
        try:
            bindings = _ldd_soname_bindings(completed.stdout)
        except ValueError as exc:
            raise RuntimeError(
                "ldd returned an unrecognized binding for owned native shared object "
                f"{owned['distribution']}:{owned['member']} at {owned['path']}"
            ) from exc
        stdout_bytes = completed.stdout.encode("utf-8")
        stderr_bytes = completed.stderr.encode("utf-8")
        if (
            len(stdout_bytes) > _NATIVE_LDD_MAX_STREAM_BYTES
            or len(stderr_bytes) > _NATIVE_LDD_MAX_STREAM_BYTES
        ):
            raise RuntimeError(
                "ldd output exceeded the evidence bound for owned native shared "
                f"object {owned['distribution']}:{owned['member']}"
            )
        record = {
            **owned,
            "ldd_returncode": completed.returncode,
            "ldd_stderr": completed.stderr,
            "ldd_stderr_lines": sorted(
                line.strip() for line in completed.stderr.splitlines() if line.strip()
            ),
            "ldd_stderr_sha256": sha256(stderr_bytes).hexdigest(),
            "ldd_stderr_utf8_bytes": len(stderr_bytes),
            "ldd_stdout": completed.stdout,
            "ldd_stdout_lines": sorted(
                line.strip() for line in completed.stdout.splitlines() if line.strip()
            ),
            "ldd_stdout_sha256": sha256(stdout_bytes).hexdigest(),
            "ldd_stdout_utf8_bytes": len(stdout_bytes),
            "resolution_scope": (
                "platform_inapplicable"
                if (owned["distribution"], owned["member"])
                in _PLATFORM_INAPPLICABLE_NATIVE_MISSING_SONAMES
                else "runtime_reachable"
            ),
            "soname_bindings": bindings,
        }
        evidence.append(record)
        missing_sonames = {
            binding["soname"]
            for binding in bindings
            if binding["resolved_path"] is None
        }
        permitted_missing_sonames = _PLATFORM_INAPPLICABLE_NATIVE_MISSING_SONAMES.get(
            (owned["distribution"], owned["member"])
        )
        if (
            completed.returncode != 0
            or bool(completed.stderr)
            or (permitted_missing_sonames is None and bool(missing_sonames))
            or (
                permitted_missing_sonames is not None
                and not missing_sonames.issubset(permitted_missing_sonames)
            )
        ):
            failures.append(record)
    if failures:
        raise RuntimeError(
            "owned native shared-object audit failed: "
            + json.dumps(
                {"failures": failures},
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return tuple(evidence)


def _canonical_native_distribution_names(
    distribution_names: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(distribution_names, (str, bytes, bytearray)) or not isinstance(
        distribution_names, Sequence
    ):
        raise TypeError("distribution_names must be a sequence")
    names: list[str] = []
    for value in distribution_names:
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is None
        ):
            raise ValueError("native distribution names must be canonical")
        names.append(value)
    if not names:
        raise ValueError("at least one native distribution is required")
    if len(set(names)) != len(names):
        raise ValueError("native distribution names must be unique")
    return tuple(sorted(names))


def _native_shared_object_member(
    value: str,
    *,
    distribution_name: str,
) -> PurePosixPath | None:
    member = PurePosixPath(value)
    if _NATIVE_SHARED_OBJECT_NAME_RE.fullmatch(member.name) is None:
        return None
    if (
        not value
        or value != member.as_posix()
        or member.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise RuntimeError(
            f"native distribution {distribution_name!r} has unsafe shared-object "
            f"member {value!r}"
        )
    return member


def _canonical_native_located_path(value: Any, *, label: str) -> Path:
    try:
        path_text = os.fspath(value)
    except TypeError as exc:
        raise RuntimeError(f"{label} is not path-like") from exc
    if not isinstance(path_text, str):
        raise RuntimeError(f"{label} must be a text path")
    path = PurePosixPath(path_text)
    if (
        not path_text
        or not path.is_absolute()
        or path_text.startswith("//")
        or path.as_posix() != path_text
        or "\\" in path_text
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or any(ord(character) < 32 for character in path_text)
    ):
        raise RuntimeError(f"{label} must be a canonical absolute path")
    return Path(path_text)


def _require_no_native_symlink_ancestors(
    path: Path,
    *,
    label: str,
    include_leaf: bool,
) -> None:
    candidate = path if include_leaf else path.parent
    while True:
        try:
            status = candidate.lstat()
        except OSError as exc:
            raise RuntimeError(f"{label} ancestor is unavailable: {candidate}") from exc
        if stat.S_ISLNK(status.st_mode):
            raise RuntimeError(f"{label} has a symlink ancestor: {candidate}")
        parent = candidate.parent
        if parent == candidate:
            return
        candidate = parent


def _ldd_reported_absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    components = value.split("/")
    if (
        not value.startswith("/")
        or value.startswith("//")
        or value.endswith("/")
        or "\\" in value
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159 for character in value
        )
        or any(component in {"", "."} for component in components[1:])
        or components[-1] == ".."
    ):
        raise ValueError(f"{label} must be a valid ldd-reported absolute path")
    depth = 0
    for component in components[1:]:
        if component == "..":
            if depth == 0:
                raise ValueError(f"{label} must be a valid ldd-reported absolute path")
            depth -= 1
        else:
            depth += 1
    return value


def _ldd_soname_bindings(stdout: str) -> list[dict[str, str | None]]:
    bindings: list[dict[str, str | None]] = []
    observed_sonames: set[str] = set()
    for raw_line in stdout.splitlines():
        stripped = raw_line.strip()
        if "=>" not in stripped:
            continue
        parts = stripped.split("=>")
        if len(parts) != 2:
            raise ValueError("ldd binding line repeats the binding separator")
        soname = parts[0].strip()
        resolution = parts[1].strip()
        if (
            not soname
            or any(character.isspace() for character in soname)
            or soname in observed_sonames
        ):
            raise ValueError("ldd binding SONAME is invalid or repeated")
        observed_sonames.add(soname)
        if resolution == "not found":
            resolved_path: str | None = None
        else:
            match = re.fullmatch(
                r"(?P<path>/.*?)(?:\s+\(0x[0-9a-fA-F]+\))?", resolution
            )
            if match is None:
                raise ValueError("ldd binding resolution is not an absolute path")
            resolved_path = _ldd_reported_absolute_path(
                match.group("path"), "ldd binding path"
            )
        bindings.append({"resolved_path": resolved_path, "soname": soname})
    return sorted(
        bindings,
        key=lambda item: (item["soname"], item["resolved_path"] or ""),
    )


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
    from document_kv_cache.gpu_qualification_sentinels import (
        _site_packages_tree_read_only,
    )

    try:
        return _site_packages_tree_read_only(
            Path(sys.prefix),
            list(site.getsitepackages()),
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _selected_bitsandbytes_native_library_path() -> Path:
    distribution_name = "bitsandbytes"
    distribution = importlib.metadata.distribution(distribution_name)
    raw_files = distribution.files
    if raw_files is None:
        raise RuntimeError("bitsandbytes has no owned file inventory")
    owned_member_count = sum(
        str(raw_member) == _SELECTED_BITSANDBYTES_NATIVE_LIBRARY_MEMBER
        for raw_member in raw_files
    )
    if owned_member_count != 1:
        raise RuntimeError(
            "bitsandbytes does not uniquely own the selected CUDA 12.9 library"
        )
    member = _native_shared_object_member(
        _SELECTED_BITSANDBYTES_NATIVE_LIBRARY_MEMBER,
        distribution_name=distribution_name,
    )
    if member is None:  # pragma: no cover - the package-owned constant is a .so
        raise RuntimeError(
            "selected bitsandbytes native library is not a shared object"
        )

    root_path = _canonical_native_located_path(
        distribution.locate_file(""),
        label="bitsandbytes native distribution root",
    )
    _require_no_native_symlink_ancestors(
        root_path,
        label="bitsandbytes native distribution root",
        include_leaf=True,
    )
    located_path = _canonical_native_located_path(
        distribution.locate_file(member),
        label="selected bitsandbytes native library located path",
    )
    expected_path = root_path.joinpath(*member.parts)
    if located_path != expected_path:
        raise RuntimeError(
            "selected bitsandbytes native library locates outside its canonical "
            f"member path: {located_path}"
        )
    _require_no_native_symlink_ancestors(
        located_path,
        label="selected bitsandbytes native library",
        include_leaf=True,
    )
    try:
        resolved_root = root_path.resolve(strict=True)
        resolved_path = located_path.resolve(strict=True)
        status = located_path.stat()
    except OSError as exc:
        raise RuntimeError(
            "selected bitsandbytes native library is unavailable"
        ) from exc
    if not resolved_root.is_dir():
        raise RuntimeError("bitsandbytes native distribution root is not a directory")
    if not stat.S_ISREG(status.st_mode) or resolved_path != located_path:
        raise RuntimeError(
            "selected bitsandbytes native library is not a canonical regular file"
        )
    if not resolved_path.is_relative_to(resolved_root):
        raise RuntimeError("selected bitsandbytes native library escapes its owner")
    return located_path


def _weight_quantizer_attestation() -> dict[str, Any]:
    """Exercise the pinned NF4/double-quant call and inspect its output state."""

    _reject_native_selector_environment()
    selected_native_path = _selected_bitsandbytes_native_library_path()
    torch = _torch()
    if getattr(getattr(torch, "version", None), "cuda", None) != "12.9":
        raise RuntimeError("bitsandbytes qualification requires PyTorch CUDA 12.9")
    import bitsandbytes.cextension as bnb_cextension  # type: ignore[import-not-found]
    import bitsandbytes.functional as bnb_functional  # type: ignore[import-not-found]
    from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

    native_library = bnb_cextension.lib
    if bnb_functional.lib is not native_library:
        raise RuntimeError(
            "bitsandbytes functional layer uses a different native library"
        )
    if getattr(native_library, "compiled_with_cuda", None) is not True:
        raise RuntimeError("bitsandbytes native library was not compiled with CUDA")
    loaded_cdll = getattr(native_library, "_lib", None)
    if not isinstance(loaded_cdll, ctypes.CDLL):
        raise RuntimeError("bitsandbytes CUDA library is not backed by ctypes.CDLL")
    loaded_name = getattr(loaded_cdll, "_name", None)
    loaded_handle = getattr(loaded_cdll, "_handle", None)
    if type(loaded_name) is not str or loaded_name != str(selected_native_path):
        raise RuntimeError(
            "bitsandbytes ctypes.CDLL did not load the selected owned CUDA 12.9 member"
        )
    if type(loaded_handle) is not int or loaded_handle <= 0:
        raise RuntimeError("bitsandbytes ctypes.CDLL has no positive native handle")
    cuda_backend = sys.modules.get("bitsandbytes.backends.cuda.ops")
    if cuda_backend is None or getattr(cuda_backend, "lib", None) is not native_library:
        raise RuntimeError(
            "bitsandbytes CUDA quantizer uses a different native library"
        )

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
    if (
        bnb_cextension.lib is not native_library
        or bnb_functional.lib is not native_library
        or getattr(cuda_backend, "lib", None) is not native_library
        or getattr(native_library, "_lib", None) is not loaded_cdll
        or getattr(loaded_cdll, "_name", None) != loaded_name
        or getattr(loaded_cdll, "_handle", None) != loaded_handle
    ):
        raise RuntimeError(
            "bitsandbytes native library identity changed during the NF4 call"
        )
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
        "loaded_native_library_member": (_SELECTED_BITSANDBYTES_NATIVE_LIBRARY_MEMBER),
        "loaded_native_library_path": str(selected_native_path),
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


def _pre_rope_handoff_layout_and_generator() -> tuple[Any, Any]:
    """Build one pre-RoPE generator already bound to its final KV layout."""

    from document_kv_cache.model_profiles import (
        QWEN3_4B_ROPE_ROTARY_DIM,
        QWEN3_4B_ROPE_THETA,
        layout_for_model,
    )
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
        shares_kv_storage=False,
    )
    generator = build_pre_rope_transformers_kv_chunk_generator()
    generator.bind_layout(layout)
    return layout, generator


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
    selected_records.sort(
        key=lambda item: (
            str(item.get("dataset", "")),
            str(item.get("example_id", "")),
        )
    )
    selected_path = work_dir / "capacity-inputs.jsonl"
    _write_jsonl(selected_path, selected_records)

    _configure_transformers_generator(pre_rope=True)
    from document_kv_cache.benchmark_handoffs import (
        generate_benchmark_handoff_bundles,
    )
    from document_kv_cache.benchmarks import DEFAULT_V1_PROMPT_TEMPLATE_VERSION
    from document_kv_cache.models import CacheGenerationMethod
    from document_kv_cache.vllm_smoke import release_handoff_generation_resources

    layout, generator = _pre_rope_handoff_layout_and_generator()
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
    request_results: list[dict[str, Any]] = []
    completion_failures: tuple[_QualificationCompletionFailure, ...] = ()
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
        except _QualificationCompletionBatchFailure as exc:
            completion_failures = exc.failures
        except _QualificationCompletionFailure as exc:
            completion_failures = (exc,)
        finally:
            terminate_process(server)
    if completion_failures:
        _raise_qualification_completion_failure(
            completion_failures,
            server_log_path=config.server_log_path,
            known_request_ids=tuple(
                _required_string(request, "request_id") for request in capacity_requests
            ),
        )
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
    expected_request_ids = tuple(
        _required_string(request, "request_id") for request in capacity_requests
    )
    layer_counts = _require_exact_connector_loads(
        _successful_connector_loads(config.connector_telemetry_path),
        expected_client_request_ids=expected_request_ids,
        label="capacity",
    )
    return {
        "attention_backend_observed": "TRITON_ATTN",
        "attention_backend_requested": "TRITON_ATTN",
        "cold_disk_evicted_file_count": evicted_files,
        "connector_loaded_layer_counts": layer_counts,
        "connector_successful_load_count": len(layer_counts),
        "fatal_error_count": 0,
        "forced_decode_tokens": GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
        "gpu_memory_utilization": gpu_memory_utilization,
        "input_tokens_per_request": input_context_tokens,
        "kv_cache_capacity_tokens": capacity,
        "max_model_len": max_model_len,
        "observed_peak_headroom_bytes": headroom,
        "observed_peak_used_memory_bytes": memory.peak_used_bytes,
        "active_request_memory_observation_count": (active_request_observation_count),
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
        DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
        DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
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
    for example_ordinal, record in enumerate(selected_records):
        dataset = str(record["dataset"])
        example_id = str(record["example_id"])
        example = _example_from_record(
            record,
            default_dataset=dataset,
            record_index=1,
            require_dataset=True,
        )
        full_prompt = build_prefill_prompt(example)
        full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
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
        request_id = _capacity_completion_request_id(example_ordinal)
        runtime_params = dict(params)
        runtime_params[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM] = request_id
        runtime_params[DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] = "logical"
        result.append(
            {
                "cache_salt": f"qualification-capacity:{dataset}:{example_id}",
                "kv_transfer_params": runtime_params,
                "prompt": full_prompt,
                "example_ordinal": example_ordinal,
                "request_id": request_id,
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
        params = _mapping(
            request.get("kv_transfer_params"), "capacity kv_transfer_params"
        )
        example_ordinal = request.get("example_ordinal")
        if type(example_ordinal) is not int or example_ordinal < 0:
            raise RuntimeError("capacity request example ordinal is invalid")
        request_id = _required_string(request, "request_id")
        context = _qualification_completion_context(
            request_id=request_id,
            sentinel_phase="capacity",
            arm_id="vanilla_prefill",
            example_ordinal=example_ordinal,
            repeat_ordinal=0,
        )
        try:
            barrier.wait(timeout=30)
            return _completion_request(
                endpoint=endpoint,
                model=model,
                prompt=_required_string(request, "prompt"),
                kv_transfer_params=params,
                cache_salt=_required_string(request, "cache_salt"),
                max_tokens=GPU_QUALIFICATION_CAPACITY_DECODE_TOKENS,
                request_id=request_id,
                sentinel_phase="capacity",
                arm_id="vanilla_prefill",
                example_ordinal=example_ordinal,
                repeat_ordinal=0,
            )
        except _QualificationCompletionFailure:
            raise
        except Exception as exc:
            raise _QualificationCompletionFailure(
                request_id=request_id,
                context=context,
                request_diagnostic={
                    "category": "unknown",
                    "exception_type": _allowlisted_exception_type(exc),
                    "kind": "client_request_error",
                },
            ) from None

    with ThreadPoolExecutor(
        max_workers=GPU_QUALIFICATION_REQUEST_PARALLELISM,
        thread_name_prefix="cachet-capacity-c4",
    ) as executor:
        futures = [executor.submit(execute, request) for request in requests]
        results: list[dict[str, Any]] = []
        failures: list[_QualificationCompletionFailure] = []
        for future in futures:
            try:
                results.append(future.result())
            except _QualificationCompletionFailure as exc:
                failures.append(exc)
        if failures:
            raise _QualificationCompletionBatchFailure(failures)
        return results


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
        raise RuntimeError(
            "vLLM server log did not attest the forced TRITON_ATTN backend"
        )


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
        if (
            type(blocks) is int
            and blocks > 0
            and type(block_size) is int
            and block_size > 0
        ):
            return blocks * block_size
    raise RuntimeError("vLLM did not expose measured GPU KV-block capacity")


def _observed_attention_backends(llm: Any) -> set[str]:
    records = llm.collective_rpc("get_model_inspection")
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
        or any(not isinstance(record, str) or not record for record in records)
    ):
        raise RuntimeError(
            "vLLM model inspection returned no nonempty worker records"
        )
    names = {
        match.group(1)
        for record in records
        for match in re.finditer(
            (
                r"(?<![A-Za-z0-9_])backend="
                r"([A-Za-z][A-Za-z0-9_]*)(?=[,\s)])"
            ),
            record,
        )
    }
    if not names:
        raise RuntimeError(
            "vLLM model inspection exposed no attention backend implementation"
        )
    normalized: set[str] = set()
    for name in names:
        upper = name.upper()
        if "TRITON" in upper:
            normalized.add("TRITON_ATTN")
        elif "FLASHINFER" in upper:
            normalized.add("FLASHINFER")
        elif (
            "FLASH_ATTN" in upper
            or "FLASHATTN" in upper
            or "FLASHATTENTION" in upper
        ):
            normalized.add("FLASH_ATTN")
        else:
            normalized.add(name)
    return normalized


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
    from document_kv_cache.models import CacheGenerationMethod
    from document_kv_cache.storage import local_path

    layout, generator = _pre_rope_handoff_layout_and_generator()
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
            _required_string(planned_job, "hardware_id") == "aws-g6e-l40s"
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
        CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV: (GPU_QUALIFICATION_MODEL_REVISION),
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
    selected_records.sort(
        key=lambda item: (
            str(item.get("dataset", "")),
            str(item.get("example_id", "")),
        )
    )
    _write_jsonl(selected_path, selected_records)

    _configure_transformers_generator(pre_rope=True)
    from document_kv_cache.benchmark_handoffs import (
        generate_benchmark_handoff_bundles,
    )
    from document_kv_cache.benchmarks import DEFAULT_V1_PROMPT_TEMPLATE_VERSION
    from document_kv_cache.models import CacheGenerationMethod
    from document_kv_cache.vllm_smoke import release_handoff_generation_resources

    layout, generator = _pre_rope_handoff_layout_and_generator()
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
    results: list[dict[str, Any]] = []
    completion_failures: tuple[_QualificationCompletionFailure, ...] = ()
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
            input_bundle_sha256=_plan_artifact_pin(plan_record, "input_bundle_sha256"),
            served_model_name=SERVED_MODEL_NAME,
        )
    except _QualificationCompletionBatchFailure as exc:
        completion_failures = exc.failures
    except _QualificationCompletionFailure as exc:
        completion_failures = (exc,)
    finally:
        terminate_process(server)
    known_request_ids = tuple(
        _matched_completion_request_id(
            example_ordinal,
            arm_id=arm_id,
            repeat_ordinal=repeat_ordinal,
        )
        for example_ordinal in range(len(selected_records))
        for arm_id in ("baseline_prefill", "vanilla_prefill")
        for repeat_ordinal in range(GPU_QUALIFICATION_DETERMINISM_REPEATS)
    )
    if completion_failures:
        _raise_qualification_completion_failure(
            completion_failures,
            server_log_path=config.server_log_path,
            known_request_ids=known_request_ids,
        )
    expected_vanilla_request_ids = tuple(
        _matched_completion_request_id(
            example_ordinal,
            arm_id="vanilla_prefill",
            repeat_ordinal=repeat_ordinal,
        )
        for example_ordinal in range(len(selected_records))
        for repeat_ordinal in range(GPU_QUALIFICATION_DETERMINISM_REPEATS)
    )
    _require_exact_connector_loads(
        _successful_connector_loads(config.connector_telemetry_path),
        expected_client_request_ids=expected_vanilla_request_ids,
        label="matched-token",
    )
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
    if argv.count("--no-enable-log-requests") != 1:
        raise RuntimeError("qualification server must disable request logging")
    if "--log-error-stack" in argv:
        raise RuntimeError("qualification server error-stack flag was already present")
    argv.append("--log-error-stack")
    if argv.count("--log-error-stack") != 1:
        raise RuntimeError("qualification server error-stack flag is not unique")
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
        DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
        DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
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
    for example_ordinal, record in enumerate(selected_records):
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
        position_hash = _integer_sequence_sha256(
            range(len(full_ids), len(full_ids) + 16)
        )
        arms: list[dict[str, Any]] = []
        for arm_id in ("baseline_prefill", "vanilla_prefill"):
            repeats: list[dict[str, Any]] = []
            for repeat in range(GPU_QUALIFICATION_DETERMINISM_REPEATS):
                request_id = _matched_completion_request_id(
                    example_ordinal,
                    arm_id=arm_id,
                    repeat_ordinal=repeat,
                )
                handoff_params: Mapping[str, Any] | None = None
                if arm_id == "vanilla_prefill":
                    runtime_params = dict(params)
                    runtime_params[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM] = request_id
                    runtime_params[DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM] = "logical"
                    handoff_params = runtime_params
                repeats.append(
                    _completion_request(
                        endpoint=endpoint,
                        model=served_model_name,
                        prompt=full_prompt,
                        kv_transfer_params=handoff_params,
                        cache_salt=(
                            f"qualification:{dataset}:{example_id}:{arm_id}:repeat-{repeat}"
                        ),
                        request_id=request_id,
                        sentinel_phase="matched_token",
                        arm_id=arm_id,
                        example_ordinal=example_ordinal,
                        repeat_ordinal=repeat,
                    )
                )
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
                    "repeat_count": GPU_QUALIFICATION_DETERMINISM_REPEATS,
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


def _capacity_completion_request_id(example_ordinal: int) -> str:
    if (
        type(example_ordinal) is not int
        or example_ordinal < 0
        or example_ordinal >= GPU_QUALIFICATION_REQUEST_PARALLELISM
    ):
        raise ValueError("capacity example ordinal is outside the closed domain")
    return f"gpuq-capacity-e{example_ordinal:02d}"


def _matched_completion_request_id(
    example_ordinal: int,
    *,
    arm_id: str,
    repeat_ordinal: int,
) -> str:
    if (
        type(example_ordinal) is not int
        or example_ordinal < 0
        or example_ordinal >= GPU_QUALIFICATION_MATCHED_EXAMPLES
    ):
        raise ValueError("matched-token example ordinal is outside the closed domain")
    if arm_id not in _QUALIFICATION_COMPLETION_ARMS:
        raise ValueError("matched-token arm is outside the closed domain")
    if (
        type(repeat_ordinal) is not int
        or repeat_ordinal < 0
        or repeat_ordinal >= GPU_QUALIFICATION_DETERMINISM_REPEATS
    ):
        raise ValueError("matched-token repeat ordinal is outside the closed domain")
    arm_label = "baseline" if arm_id == "baseline_prefill" else "vanilla"
    return f"gpuq-matched-e{example_ordinal:02d}-{arm_label}-r{repeat_ordinal}"


def _completion_request(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    kv_transfer_params: Mapping[str, Any] | None,
    cache_salt: str,
    max_tokens: int = 16,
    request_id: str,
    sentinel_phase: str,
    arm_id: str,
    example_ordinal: int,
    repeat_ordinal: int,
) -> dict[str, Any]:
    context = _qualification_completion_context(
        request_id=request_id,
        sentinel_phase=sentinel_phase,
        arm_id=arm_id,
        example_ordinal=example_ordinal,
        repeat_ordinal=repeat_ordinal,
    )
    body: dict[str, Any] = {
        "add_special_tokens": False,
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 17,
        "ignore_eos": True,
        "logprobs": 1,
        "return_token_ids": True,
        "cache_salt": cache_salt,
        "request_id": request_id,
    }
    if kv_transfer_params is not None:
        body["kv_transfer_params"] = dict(kv_transfer_params)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response_evidence: dict[str, Any] = {}

    def fail_response_contract(
        kind: str,
        *,
        exception: BaseException | None = None,
        counts: Mapping[str, int] | None = None,
    ) -> NoReturn:
        diagnostic: dict[str, Any] = {
            **response_evidence,
            "category": "unknown",
            "kind": kind,
        }
        if exception is not None:
            diagnostic["exception_type"] = _allowlisted_exception_type(exception)
        if counts:
            diagnostic["counts"] = dict(counts)
        raise _QualificationCompletionFailure(
            request_id=request_id,
            context=context,
            request_diagnostic=diagnostic,
        ) from None

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            try:
                observed = response.read(
                    _QUALIFICATION_COMPLETION_RESPONSE_MAX_BYTES + 1
                )
            except Exception as exc:
                fail_response_contract("response_read_error", exception=exc)
        captured = observed[:_QUALIFICATION_COMPLETION_RESPONSE_MAX_BYTES]
        response_evidence = {
            "captured_byte_count": len(captured),
            "captured_sha256": sha256(captured).hexdigest(),
            "captured_truncated": len(observed) > len(captured),
        }
        if len(observed) > len(captured):
            fail_response_contract("response_too_large")
        try:
            record = json.loads(captured.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail_response_contract("response_decode_error", exception=exc)
    except urllib.error.HTTPError as exc:
        diagnostic = _bounded_http_error_diagnostic(exc)
        raise _QualificationCompletionFailure(
            request_id=request_id,
            context=context,
            request_diagnostic=diagnostic,
        ) from None
    except urllib.error.URLError as exc:
        raise _QualificationCompletionFailure(
            request_id=request_id,
            context=context,
            request_diagnostic={
                "category": "unknown",
                "kind": "transport_error",
                "transport_error_type": _allowlisted_exception_type(exc.reason),
            },
        ) from None
    except TimeoutError as exc:
        raise _QualificationCompletionFailure(
            request_id=request_id,
            context=context,
            request_diagnostic={
                "category": "unknown",
                "kind": "timeout",
                "transport_error_type": _allowlisted_exception_type(exc),
            },
        ) from None
    except OSError as exc:
        raise _QualificationCompletionFailure(
            request_id=request_id,
            context=context,
            request_diagnostic={
                "category": "unknown",
                "kind": "transport_error",
                "transport_error_type": _allowlisted_exception_type(exc),
            },
        ) from None
    choices = record.get("choices") if isinstance(record, Mapping) else None
    if not isinstance(choices, list) or len(choices) != 1:
        fail_response_contract(
            "response_contract_error",
            counts={"choice_count": len(choices) if isinstance(choices, list) else 0},
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        fail_response_contract("response_contract_error")
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
        fail_response_contract(
            "response_contract_error",
            counts={
                "expected_token_id_count": max_tokens,
                "token_id_count": len(token_ids) if isinstance(token_ids, list) else 0,
            },
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
        fail_response_contract(
            "response_contract_error",
            counts={
                "token_id_count": len(token_ids),
                "token_logprob_count": (
                    len(token_logprobs) if isinstance(token_logprobs, list) else 0
                ),
            },
        )
    return {
        "token_ids": token_ids,
        "token_logprobs": [float(item) for item in token_logprobs],
    }


def _qualification_completion_context(
    *,
    request_id: str,
    sentinel_phase: str,
    arm_id: str,
    example_ordinal: int,
    repeat_ordinal: int,
) -> dict[str, Any]:
    if sentinel_phase not in _QUALIFICATION_COMPLETION_PHASES:
        raise ValueError("qualification completion phase is outside the closed domain")
    if arm_id not in _QUALIFICATION_COMPLETION_ARMS:
        raise ValueError("qualification completion arm is outside the closed domain")
    if sentinel_phase == "capacity":
        if arm_id != "vanilla_prefill" or repeat_ordinal != 0:
            raise ValueError("capacity completion context is invalid")
        expected_request_id = _capacity_completion_request_id(example_ordinal)
    else:
        expected_request_id = _matched_completion_request_id(
            example_ordinal,
            arm_id=arm_id,
            repeat_ordinal=repeat_ordinal,
        )
    if request_id != expected_request_id:
        raise ValueError("qualification completion request identity drift")
    return {
        "arm_id": arm_id,
        "example_ordinal": example_ordinal,
        "repeat_ordinal": repeat_ordinal,
        "sentinel_phase": sentinel_phase,
    }


def _bounded_http_error_diagnostic(
    error: urllib.error.HTTPError,
) -> dict[str, Any]:
    try:
        try:
            observed = error.read(_QUALIFICATION_HTTP_ERROR_MAX_BYTES + 1)
        except Exception as exc:
            return {
                "body_read_error_type": _allowlisted_exception_type(exc),
                "category": "unknown",
                "http_status": int(error.code),
                "kind": "http_error_body_unavailable",
            }
    finally:
        error.close()
    captured = observed[:_QUALIFICATION_HTTP_ERROR_MAX_BYTES]
    diagnostic: dict[str, Any] = {
        "captured_byte_count": len(captured),
        "captured_sha256": sha256(captured).hexdigest(),
        "captured_truncated": len(observed) > len(captured),
        "http_status": int(error.code),
        "kind": "http_error",
    }
    diagnostic.update(_qualification_error_evidence(captured))
    try:
        decoded = json.loads(captured.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, Mapping):
        error_record = decoded.get("error")
        api_error_type = (
            error_record.get("type")
            if isinstance(error_record, Mapping)
            else decoded.get("type")
        )
        if api_error_type in {
            "BadRequestError",
            "EngineDeadError",
            "Internal Server Error",
            "InternalServerError",
            "RuntimeError",
            "ValueError",
        }:
            diagnostic["api_error_type"] = api_error_type
    return diagnostic


def _qualification_error_evidence(content: bytes) -> dict[str, Any]:
    text = content.decode("utf-8", errors="replace")
    lower = text.lower()
    if "fewer than the cached prefix length" in lower:
        category = "token_contract_shortfall"
    elif "token contract" in lower and any(
        marker in lower for marker in ("differ", "mismatch", "does not match")
    ):
        category = "token_contract_mismatch"
    elif "num_computed_tokens" in lower and "num_tokens" in lower:
        category = "scheduler_visible_token_invariant"
    elif "unknown" in lower and "kv" in lower and "layer" in lower:
        category = "unknown_kv_layer"
    elif any(
        marker in lower
        for marker in ("cuda out of memory", "outofmemoryerror", "cudaerroroutofmemory")
    ):
        category = "cuda_oom"
    elif "triton" in lower and any(
        marker in lower for marker in ("compile", "compilation", "kernel launch")
    ):
        category = "triton_compile_or_launch"
    elif "enginedeaderror" in lower or (
        "enginecore" in lower
        and any(marker in lower for marker in ("dead", "failed", "error"))
    ):
        category = "engine_dead"
    else:
        category = "unknown"
    evidence: dict[str, Any] = {"category": category}
    exception_type = _allowlisted_exception_type_from_text(text)
    if exception_type != "unknown":
        evidence["exception_type"] = exception_type
    counts: dict[str, int] = {}
    shortfall = re.search(
        r"exposes\s+([0-9]+)\s+token ids,\s+fewer than\s+the cached prefix length\s+([0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if shortfall is not None:
        counts["visible_token_count"] = int(shortfall.group(1))
        counts["cached_prefix_token_count"] = int(shortfall.group(2))
    for field_name in ("num_computed_tokens", "num_tokens"):
        match = re.search(
            rf"\b{field_name}\b\s*(?:=|:)\s*([0-9]+)",
            text,
            flags=re.IGNORECASE,
        )
        if match is not None:
            counts[field_name] = int(match.group(1))
    if counts:
        evidence["counts"] = counts
    return evidence


def _allowlisted_exception_type(value: object) -> str:
    return _allowlisted_exception_type_from_text(type(value).__name__)


def _allowlisted_exception_type_from_text(text: str) -> str:
    for name in (
        "EngineDeadError",
        "JSONDecodeError",
        "UnicodeDecodeError",
        "IncompleteRead",
        "FileNotFoundError",
        "PermissionError",
        "IsADirectoryError",
        "RemoteDisconnected",
        "OutOfMemoryError",
        "AssertionError",
        "BrokenBarrierError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "TimeoutError",
        "HTTPError",
        "RuntimeError",
        "ValueError",
        "OSError",
    ):
        if re.search(rf"\b{re.escape(name)}\b", text) is not None:
            return name
    return "unknown"


def _server_log_failure_diagnostic(
    path: Path,
    *,
    known_request_ids: Sequence[str],
) -> dict[str, Any]:
    request_ids = tuple(sorted(set(known_request_ids)))
    if len(request_ids) != len(known_request_ids) or any(
        not request_id or "\n" in request_id for request_id in request_ids
    ):
        raise ValueError("qualification diagnostic request identities are invalid")
    needles = {request_id: request_id.encode("utf-8") for request_id in request_ids}
    overlap_size = max((len(needle) for needle in needles.values()), default=1) - 1
    overlap = b""
    tail = bytearray()
    digest = sha256()
    byte_count = 0
    seen: set[str] = set()
    descriptor = -1
    try:
        _require_no_native_symlink_ancestors(
            path,
            label="qualification server log",
            include_leaf=False,
        )
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return {
                "available": False,
                "category": "unknown",
                "io_error_type": "unknown",
                "unavailable_reason": "non_regular_file",
            }
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                byte_count += len(chunk)
                digest.update(chunk)
                searchable = overlap + chunk
                for request_id, needle in needles.items():
                    if request_id not in seen and needle in searchable:
                        seen.add(request_id)
                overlap = searchable[-overlap_size:] if overlap_size else b""
                tail.extend(chunk)
                if len(tail) > _QUALIFICATION_SERVER_LOG_TAIL_MAX_BYTES:
                    del tail[:-_QUALIFICATION_SERVER_LOG_TAIL_MAX_BYTES]
    except (OSError, RuntimeError) as exc:
        return {
            "available": False,
            "category": "unknown",
            "io_error_type": _allowlisted_exception_type(exc),
            "unavailable_reason": "io_error",
        }
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    diagnostic: dict[str, Any] = {
        "available": True,
        "byte_count": byte_count,
        "request_ids_seen": sorted(seen),
        "sha256": digest.hexdigest(),
        "tail_inspected_byte_count": len(tail),
        "tail_truncated": byte_count > len(tail),
    }
    diagnostic.update(_qualification_error_evidence(bytes(tail)))
    return diagnostic


def _raise_qualification_completion_failure(
    failures: Sequence[_QualificationCompletionFailure],
    *,
    server_log_path: Path,
    known_request_ids: Sequence[str],
) -> None:
    if not failures:
        raise ValueError("qualification completion failure set is empty")
    known_id_set = set(known_request_ids)
    if len(known_id_set) != len(known_request_ids):
        raise ValueError("qualification diagnostic request identities are duplicated")
    closed_context_keys = {
        "arm_id",
        "example_ordinal",
        "repeat_ordinal",
        "request_id",
        "sentinel_phase",
    }
    for failure in failures:
        if failure.request_id not in known_id_set:
            raise ValueError("qualification failure request identity is not known")
        if set(failure.request_diagnostic).intersection(closed_context_keys):
            raise ValueError("qualification request diagnostic overlaps its context")
        sentinel_phase = failure.context.get("sentinel_phase")
        arm_id = failure.context.get("arm_id")
        example_ordinal = failure.context.get("example_ordinal")
        repeat_ordinal = failure.context.get("repeat_ordinal")
        if (
            not isinstance(sentinel_phase, str)
            or not isinstance(arm_id, str)
            or type(example_ordinal) is not int
            or type(repeat_ordinal) is not int
        ):
            raise ValueError("qualification failure request context is invalid")
        expected_context = _qualification_completion_context(
            request_id=failure.request_id,
            sentinel_phase=sentinel_phase,
            arm_id=arm_id,
            example_ordinal=example_ordinal,
            repeat_ordinal=repeat_ordinal,
        )
        if failure.context != expected_context:
            raise ValueError("qualification failure request context differs")
    ordered = sorted(
        failures,
        key=lambda failure: (
            str(failure.context.get("sentinel_phase")),
            int(failure.context.get("example_ordinal", -1)),
            str(failure.context.get("arm_id")),
            int(failure.context.get("repeat_ordinal", -1)),
        ),
    )
    request_diagnostics = [
        {
            **failure.request_diagnostic,
            **failure.context,
            "request_id": failure.request_id,
        }
        for failure in ordered
    ]
    server_log = _server_log_failure_diagnostic(
        server_log_path,
        known_request_ids=known_request_ids,
    )
    categories = [
        str(record.get("category", "unknown"))
        for record in (*request_diagnostics, server_log)
    ]
    category = next(
        (candidate for candidate in categories if candidate != "unknown"),
        "unknown",
    )
    record = {
        "category": category,
        "record_type": "cachet.gpu_qualification_completion_failure.v1",
        "requests": request_diagnostics,
        "schema_version": 1,
        "server_log": server_log,
    }
    payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
    message = f"qualification completion failure: {payload}"
    if "\n" in message or len(message.encode("utf-8")) >= (
        _QUALIFICATION_FAILURE_DIAGNOSTIC_MAX_BYTES
    ):
        raise RuntimeError(
            "qualification completion failure diagnostic exceeded its closed bound"
        ) from None
    raise RuntimeError(message) from None


def _successful_connector_loads(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("connector telemetry was not written")
    records: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if (
            isinstance(value, Mapping)
            and value.get("record_type") == "document_kv.vllm_native_provider_load.v1"
            and value.get("event") == "load_request"
            and value.get("success") is True
        ):
            records.append(value)
    if not records:
        raise RuntimeError("connector telemetry contains no successful loads")
    return records


def _require_exact_connector_loads(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_client_request_ids: Sequence[str],
    label: str,
) -> list[int]:
    expected_ids = tuple(sorted(expected_client_request_ids))
    expected_count = 4 if label == "capacity" else 8 if label == "matched-token" else 0
    if (
        expected_count == 0
        or len(expected_ids) != expected_count
        or len(set(expected_ids)) != expected_count
        or len(records) != expected_count
    ):
        raise RuntimeError(f"{label} connector load closure is not exact")
    expected_pairs = {
        (request_id, f"cmpl-{request_id}-0") for request_id in expected_ids
    }
    observed_pairs: list[tuple[str, str]] = []
    layers_by_client_id: dict[str, int] = {}
    for record in records:
        benchmark_request_id = record.get("benchmark_request_id")
        runtime_request_id = record.get("request_id")
        if not isinstance(benchmark_request_id, str) or not isinstance(
            runtime_request_id, str
        ):
            raise RuntimeError(f"{label} connector load identity is invalid")
        observed_pairs.append((benchmark_request_id, runtime_request_id))
        counts = record.get("counts")
        layers = counts.get("layers_loaded") if isinstance(counts, Mapping) else None
        if layers != GPU_QUALIFICATION_MODEL_LAYER_COUNT:
            raise RuntimeError(
                f"{label} connector load did not inject all model layers"
            )
        if benchmark_request_id in layers_by_client_id:
            raise RuntimeError(f"{label} connector load identity is duplicated")
        layers_by_client_id[benchmark_request_id] = int(layers)
    if (
        len(set(observed_pairs)) != expected_count
        or set(observed_pairs) != expected_pairs
    ):
        raise RuntimeError(f"{label} connector load identity closure differs")
    return [layers_by_client_id[request_id] for request_id in expected_ids]


def _bucket_dataset_paths(input_bundle: Path, length: int) -> list[Path]:
    directory = input_bundle / str(length)
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError(f"input bundle is missing the {length} bucket")
    paths = [
        directory / f"{dataset}.jsonl" for dataset in GPU_QUALIFICATION_INPUT_DATASETS
    ]
    observed = {path.name for path in directory.glob("*.jsonl")}
    expected = {path.name for path in paths}
    if observed != expected or any(
        not path.is_file() or path.is_symlink() for path in paths
    ):
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
    import torch

    return torch


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
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
