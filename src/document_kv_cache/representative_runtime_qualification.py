"""A live BF16 native-transport gate, run before representative measurements.

The small deterministic tensor problem checks GPU handoff transport, absolute
RoPE orientation, and vLLM Triton attention. It does not load a model or certify
full-context logits, generation semantics, benchmark scores, or launch authority.
The caller must verify the installed source/native closure first and bind this
record to the genuine task receipt and staged immutable input set afterward.
Importing this module does not import torch or perform any GPU work.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
from typing import Any, cast

RECORD_TYPE = "cachet.representative_runtime_qualification.v1"
QUALIFICATION_SCOPE = "synthetic_bf16_native_transport_rope_and_attention_only"
LAYER_COUNT = 36
TOKEN_COUNT = 33
KV_HEADS = 8
QUERY_HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE = 16
ROPE_THETA = 5_000_000.0
KEY_MAX_ABS_ERROR = 1.0 / 128
ATTENTION_MAX_ABS_ERROR = 1.0 / 64
_SHA_FIELDS = (
    "source_tree_sha256",
    "runner_sha256",
    "package_wheel_sha256",
    "native_runtime_closure_sha256",
    "patched_vllm_wheel_sha256",
    "patched_flashinfer_wheel_sha256",
    "runtime_lock_sha256",
    "prepared_input_bundle_sha256",
    "bundle_closed_record_sha256",
    "stage_attestation_closed_record_sha256",
)
_BINDING_FIELDS = frozenset(
    (
        *_SHA_FIELDS,
        "source_commit",
        "benchmark_id",
        "runtime_id",
        "arm_id",
        "context_tokens",
    )
)
_ARMS = frozenset(
    (
        "baseline_prefill",
        "document_kv_cache:full_prefix_prefill",
        "document_kv_cache:vanilla_prefill",
    )
)
_GEOMETRY = {
    "layers": LAYER_COUNT,
    "tokens": TOKEN_COUNT,
    "kv_heads": KV_HEADS,
    "query_heads": QUERY_HEADS,
    "head_dim": HEAD_DIM,
    "block_size": BLOCK_SIZE,
    "physical_block_order": [2, 0, 3],
    "unused_physical_block": 1,
    "dtype": "bfloat16",
    "layout": "B_H_N_2D",
    "rope_theta": ROPE_THETA,
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _closed_hash(record: Mapping[str, Any]) -> str:
    return sha256(_json_bytes({**record, "closed_record_sha256": ""})).hexdigest()


def _sha(value: object, name: str, *, length: int = 64) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None
    ):
        raise ValueError(f"invalid {name}")


def _bindings(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BINDING_FIELDS:
        raise ValueError("qualification binding fields drift")
    for name in _SHA_FIELDS:
        _sha(value[name], name)
    _sha(value["source_commit"], "source_commit", length=40)
    for name in ("benchmark_id", "runtime_id"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError(f"invalid {name}")
    if value["arm_id"] not in _ARMS:
        raise ValueError("qualification arm drift")
    if type(value["context_tokens"]) is not int or value["context_tokens"] not in (
        8192,
        16384,
    ):
        raise ValueError("qualification context drift")
    return dict(value)


def _number(value: object, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {name}")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < 0
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"invalid {name}")
    return result


def validate_representative_runtime_qualification_record(
    record: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a closed task-bound result; never issue execution authority."""
    expected = _bindings(expected_bindings)
    fields = {
        "record_type",
        "schema_version",
        "scope",
        "bindings",
        "geometry",
        "runtime",
        "cases",
        "started_epoch_ns",
        "finished_epoch_ns",
        "model_forward_executed",
        "full_context_semantic_parity_claimed",
        "closed_record_sha256",
    }
    if not isinstance(record, Mapping) or set(record) != fields:
        raise ValueError("qualification record fields drift")
    if (
        record["record_type"] != RECORD_TYPE
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
    ):
        raise ValueError("qualification schema drift")
    if record["scope"] != QUALIFICATION_SCOPE:
        raise ValueError("qualification scope drift")
    if (
        record["model_forward_executed"] is not False
        or record["full_context_semantic_parity_claimed"] is not False
    ):
        raise ValueError("qualification cannot claim model semantic parity")
    if _bindings(record["bindings"]) != expected:
        raise ValueError("qualification task/source/staging binding drift")
    if _json_bytes(record["geometry"]) != _json_bytes(_GEOMETRY):
        raise ValueError("qualification geometry drift")
    for name in ("started_epoch_ns", "finished_epoch_ns"):
        if type(record[name]) is not int or record[name] <= 0:
            raise ValueError("qualification timestamp invalid")
    if record["finished_epoch_ns"] <= record["started_epoch_ns"]:
        raise ValueError("qualification timestamps are not ordered")
    _validate_runtime(record["runtime"])
    cases = record["cases"]
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("qualification requires both BF16 position encodings")
    for case, encoding in zip(cases, ("stored_post_rope", "pre_rope"), strict=True):
        _validate_case(case, encoding)
    _sha(record["closed_record_sha256"], "closed_record_sha256")
    if record["closed_record_sha256"] != _closed_hash(record):
        raise ValueError("qualification record closure drift")
    return cast(dict[str, Any], json.loads(_json_bytes(record)))


def _validate_runtime(runtime: object) -> None:
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python",
        "torch",
        "vllm",
        "triton",
        "cuda",
        "device_type",
        "device_count",
        "device_name",
        "device_capability",
        "bf16_supported",
        "provider_source_sha256",
        "attention_source_sha256",
    }:
        raise ValueError("qualification runtime fields drift")
    if (
        not isinstance(runtime["python"], str)
        or not runtime["python"].startswith("3.11.")
        or runtime["torch"] != "2.13.0+cu129"
        or runtime["vllm"] != "0.27.1+cu129"
        or runtime["triton"] != "3.7.1"
        or runtime["cuda"] != "12.9"
        or runtime["device_type"] != "cuda"
        or type(runtime["device_count"]) is not int
        or runtime["device_count"] != 1
        or runtime["bf16_supported"] is not True
    ):
        raise ValueError("qualification requires the pinned CUDA BF16 runtime")
    if runtime["device_name"] not in {"NVIDIA L4", "NVIDIA A10G"}:
        raise ValueError("qualification GPU is outside the serving matrix")
    expected_capability = [8, 9] if runtime["device_name"] == "NVIDIA L4" else [8, 6]
    if runtime["device_capability"] != expected_capability:
        raise ValueError("qualification GPU capability drift")
    for name in ("provider_source_sha256", "attention_source_sha256"):
        _sha(runtime[name], name)


def _validate_case(case: object, encoding: str) -> None:
    if not isinstance(case, Mapping) or set(case) != {
        "key_position_encoding",
        "persisted_payload_sha256",
        "loaded_payload_sha256",
        "persisted_payload_bytes",
        "loaded_layers",
        "injected_layers",
        "segment_count",
        "provider_load_calls",
        "value_mismatch_count",
        "unused_slot_mismatch_count",
        "key_max_abs_error",
        "attention_max_abs_error_by_layer",
        "attention_launches",
        "unrotated_key_negative_control_max_abs_delta",
        "local_position_reset_negative_control_max_abs_delta",
    }:
        raise ValueError("qualification case fields drift")
    if case["key_position_encoding"] != encoding:
        raise ValueError("qualification position encoding coverage drift")
    for name in ("persisted_payload_sha256", "loaded_payload_sha256"):
        _sha(case[name], name)
    if case["persisted_payload_sha256"] != case["loaded_payload_sha256"]:
        raise ValueError("qualification persisted/loaded payload mismatch")
    counts = {
        "persisted_payload_bytes": TOKEN_COUNT
        * LAYER_COUNT
        * 2
        * KV_HEADS
        * HEAD_DIM
        * 2,
        "loaded_layers": LAYER_COUNT,
        "injected_layers": LAYER_COUNT,
        "segment_count": 2 if encoding == "pre_rope" else 1,
        "provider_load_calls": 2 if encoding == "pre_rope" else 1,
        "value_mismatch_count": 0,
        "unused_slot_mismatch_count": 0,
        "attention_launches": LAYER_COUNT,
    }
    for name, expected in counts.items():
        if type(case[name]) is not int or case[name] != expected:
            raise ValueError(f"qualification {name} drift")
    _number(
        case["key_max_abs_error"],
        "key_max_abs_error",
        maximum=KEY_MAX_ABS_ERROR if encoding == "pre_rope" else 0.0,
    )
    errors = case["attention_max_abs_error_by_layer"]
    if not isinstance(errors, list) or len(errors) != LAYER_COUNT:
        raise ValueError("qualification attention layer coverage drift")
    for error in errors:
        _number(error, "attention error", maximum=ATTENTION_MAX_ABS_ERROR)
    for name in (
        "unrotated_key_negative_control_max_abs_delta",
        "local_position_reset_negative_control_max_abs_delta",
    ):
        if _number(case[name], name) < 0.25:
            raise ValueError("qualification RoPE negative control is degenerate")


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    # The probe is only 4.9 MB; this keeps CPU unit coverage independent of the
    # optional NumPy dependency while preserving BF16 bytes exactly.
    return bytes(
        tensor.detach().contiguous().view(torch.uint8).cpu().flatten().tolist()
    )


def _reference_rope(torch: Any, keys: Any, start: int = 0) -> Any:
    """Independent rotate-half reference; no production RoPE helper is called."""
    half = HEAD_DIM // 2
    angles = [
        [(position + start) / (ROPE_THETA ** (2 * i / HEAD_DIM)) for i in range(half)]
        for position in range(int(keys.shape[0]))
    ]
    cosine = torch.tensor(
        [[math.cos(x) for x in row] for row in angles],
        device=keys.device,
        dtype=torch.float32,
    )[:, None, :]
    sine = torch.tensor(
        [[math.sin(x) for x in row] for row in angles],
        device=keys.device,
        dtype=torch.float32,
    )[:, None, :]
    left, right = keys.float()[..., :half], keys.float()[..., half:]
    return torch.cat(
        (left * cosine - right * sine, right * cosine + left * sine), dim=-1
    ).to(torch.bfloat16)


def _native_roundtrip(
    torch: Any, root: Path, *, pre_rope: bool, device: str
) -> tuple[dict[str, Any], list[Any], Any, Any]:
    """Exercise the real public provider lifecycle and persisted handoff files."""
    from document_kv_cache.engine import EngineReadyRequest
    from document_kv_cache.engine_adapters import (
        build_engine_adapter_request,
        build_engine_kv_connector_actions,
        build_engine_kv_injection_plan,
        engine_kv_connector_actions_to_record,
        vllm_adapter_spec,
    )
    from document_kv_cache.engine_probe import write_engine_adapter_handoff_bundle
    from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
    from document_kv_cache.methods import method_spec
    from vllm_kv_injection.block_mapping import BlockSpan
    from vllm_kv_injection.vllm_native_provider import (
        DocumentKVConnectorMetadata,
        DocumentKVLoadRequest,
        DocumentKVNativeProvider,
    )

    encoding = "pre_rope" if pre_rope else "stored_post_rope"
    case_root = root / encoding
    case_root.mkdir()
    # Small bounded, exactly representable BF16 inputs vary across every axis.
    raw = (
        (
            torch.arange(
                TOKEN_COUNT * LAYER_COUNT * 2 * KV_HEADS * HEAD_DIM,
                device=device,
                dtype=torch.int64,
            )
            * 17
        )
        % 31
        - 15
    ).to(torch.bfloat16) / 16
    raw = raw.reshape(TOKEN_COUNT, LAYER_COUNT, 2, KV_HEADS, HEAD_DIM)
    reference_keys = torch.stack(
        [_reference_rope(torch, raw[:, i, 0]) for i in range(LAYER_COUNT)], dim=1
    )
    reference_values = raw[:, :, 1].clone()
    negative_unrotated = float(
        (reference_keys.float() - raw[:, :, 0].float()).abs().max().item()
    )
    local_reset = _reference_rope(torch, raw[17:, 0, 0])
    negative_reset = float(
        (reference_keys[17:, 0].float() - local_reset.float()).abs().max().item()
    )
    stored = raw.clone()
    if not pre_rope:
        stored[:, :, 0] = reference_keys
    payload = _tensor_bytes(stored)
    bytes_per_token = LAYER_COUNT * 2 * KV_HEADS * HEAD_DIM * 2
    layout = KVLayout(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        lora_id="base",
        layout_version="representative-bf16-qualification-v1",
        dtype="bfloat16",
        num_layers=LAYER_COUNT,
        block_size=BLOCK_SIZE,
        bytes_per_token=bytes_per_token,
        num_query_heads=QUERY_HEADS,
        num_kv_heads=KV_HEADS,
        head_size=HEAD_DIM,
        kv_stride_bytes=HEAD_DIM * 2,
        storage_layout="separate_key_value",
        pre_rope=pre_rope,
        rope_theta=ROPE_THETA if pre_rope else None,
        rope_rotary_dim=HEAD_DIM if pre_rope else None,
        key_position_encoding=encoding,
    )
    spans = ((0, 17), (17, 16)) if pre_rope else ((0, TOKEN_COUNT),)
    segments = tuple(
        KVSegment(
            f"synthetic-{i}",
            "document_chunk",
            f"segment-{i}",
            start,
            count,
            start * bytes_per_token,
            count * bytes_per_token,
        )
        for i, (start, count) in enumerate(spans)
    )
    method = "vanilla_prefill" if pre_rope else "full_prefix_prefill"
    handle = KVCacheHandle(
        request_id=encoding,
        handle_uri=f"document-kv://{encoding}",
        layout=layout,
        segments=segments,
        total_tokens=TOKEN_COUNT,
        total_bytes=len(payload),
        cache_method=method,
        payload_checksum=sha256(payload).hexdigest(),
    )
    ready = EngineReadyRequest(
        handle=handle,
        payload=payload,
        estimated_gpu_bytes=len(payload),
        reuse_plan=method_spec(method).reuse_plan(),
    )
    adapter = build_engine_adapter_request(ready, spec=vllm_adapter_spec())
    handoff_path, payload_path = write_engine_adapter_handoff_bundle(
        adapter, case_root / "handoff.json", payload_uri=str(case_root / "payload.bin")
    )
    loaded = payload_path.read_bytes()
    if loaded != payload:
        raise RuntimeError("persisted BF16 handoff bytes changed")
    plan = build_engine_kv_injection_plan(
        json.loads(handoff_path.read_bytes()), expected_backend="vllm"
    )
    actions = build_engine_kv_connector_actions(plan, loaded)
    actions_record = engine_kv_connector_actions_to_record(actions)
    provider = DocumentKVNativeProvider(payload_cache_max_bytes=0)
    pages = [
        torch.full(
            (4, KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM),
            -7,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(LAYER_COUNT)
    ]
    provider.register_kv_caches(
        {f"model.layers.{i}.self_attn.attn": page for i, page in enumerate(pages)}
    )
    block_order = (2, 0, 3)
    for start, count in spans:
        blocks = []
        local = 0
        while local < count:
            logical = start + local
            take = min(count - local, BLOCK_SIZE - logical % BLOCK_SIZE)
            blocks.append(
                BlockSpan(
                    block_order[logical // BLOCK_SIZE],
                    local,
                    take,
                    logical % BLOCK_SIZE,
                )
            )
            local += take
        load = DocumentKVLoadRequest(
            request_id=encoding,
            actions_record=actions_record,
            payload=None,
            payload_uri=str(payload_path),
            blocks=tuple(blocks),
            source_token_start=start,
            token_count=count,
        )
        provider.bind_connector_metadata(DocumentKVConnectorMetadata(loads=(load,)))
        provider.start_load_kv(None)
    if device == "cuda":
        torch.cuda.synchronize()
    slots = torch.tensor(
        [
            block_order[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
            for i in range(TOKEN_COUNT)
        ],
        device=device,
        dtype=torch.long,
    )
    mask = torch.ones(4 * BLOCK_SIZE, device=device, dtype=torch.bool)
    mask[slots] = False
    key_error, value_mismatches, unused_mismatches = 0.0, 0, 0
    for i, page in enumerate(pages):
        logical_page = page.permute(0, 2, 1, 3).reshape(
            4 * BLOCK_SIZE, KV_HEADS, 2 * HEAD_DIM
        )
        actual = logical_page[slots]
        key_error = max(
            key_error,
            float(
                (actual[..., :HEAD_DIM].float() - reference_keys[:, i].float())
                .abs()
                .max()
                .item()
            ),
        )
        value_mismatches += int(
            (actual[..., HEAD_DIM:] != reference_values[:, i]).sum().item()
        )
        unused_mismatches += int((logical_page[mask] != -7).sum().item())
    case = {
        "key_position_encoding": encoding,
        "persisted_payload_sha256": sha256(payload).hexdigest(),
        "loaded_payload_sha256": sha256(loaded).hexdigest(),
        "persisted_payload_bytes": len(payload),
        "loaded_layers": LAYER_COUNT,
        "injected_layers": len(pages),
        "segment_count": len(segments),
        "provider_load_calls": len(spans),
        "value_mismatch_count": value_mismatches,
        "unused_slot_mismatch_count": unused_mismatches,
        "key_max_abs_error": key_error,
        "unrotated_key_negative_control_max_abs_delta": negative_unrotated,
        "local_position_reset_negative_control_max_abs_delta": negative_reset,
    }
    return case, pages, reference_keys, reference_values


def _attention_reference(torch: Any, query: Any, keys: Any, values: Any) -> Any:
    keys = keys.repeat_interleave(QUERY_HEADS // KV_HEADS, dim=1).float()
    values = values.repeat_interleave(QUERY_HEADS // KV_HEADS, dim=1).float()
    scores = torch.einsum("qhd,thd->qht", query.float(), keys) * HEAD_DIM**-0.5
    return torch.einsum("qht,thd->qhd", torch.softmax(scores, dim=-1), values)


def run_representative_runtime_qualification(
    *,
    output_path: Path,
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Run on the genuine single-GPU serving runtime, then write a fresh result."""
    bindings = _bindings(expected_bindings)
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("qualification output must be fresh")
    if not output_path.parent.is_dir() or output_path.parent.is_symlink():
        raise ValueError("qualification output parent must be a real directory")
    started = time.time_ns()
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("qualification requires one real CUDA BF16 device")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("qualification requires the pinned CPython 3.11 runtime")
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # type: ignore[import-not-found]
    from vllm.v1.kv_cache_interface import KVQuantMode  # type: ignore[import-not-found]
    import vllm_kv_injection.vllm_native_provider as provider_module

    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "vllm": importlib.metadata.version("vllm"),
        "triton": importlib.metadata.version("triton"),
        "cuda": torch.version.cuda,
        "device_type": "cuda",
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": True,
        "provider_source_sha256": sha256(
            Path(str(provider_module.__file__)).read_bytes()
        ).hexdigest(),
        "attention_source_sha256": sha256(
            Path(unified_attention.__code__.co_filename).read_bytes()
        ).hexdigest(),
    }
    _validate_runtime(runtime)
    cases = []
    # The private synthetic handoffs live beside the record, separate from all
    # measured staged artifacts. A failed gate leaves its diagnostics for review.
    work_dir = Path(
        tempfile.mkdtemp(prefix="runtime-qualification-", dir=output_path.parent)
    )
    for pre_rope in (False, True):
        case, pages, keys, values = _native_roundtrip(
            torch, work_dir, pre_rope=pre_rope, device="cuda"
        )
        errors = []
        for i, page in enumerate(pages):
            query = (
                keys[-1:, i]
                .repeat_interleave(QUERY_HEADS // KV_HEADS, dim=1)
                .contiguous()
            )
            output = torch.empty_like(query)
            unified_attention(
                q=query,
                k=page[..., :HEAD_DIM].permute(0, 2, 1, 3),
                v=page[..., HEAD_DIM:].permute(0, 2, 1, 3),
                out=output,
                cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                max_seqlen_q=1,
                seqused_k=torch.tensor([TOKEN_COUNT], device="cuda", dtype=torch.int32),
                max_seqlen_k=TOKEN_COUNT,
                softmax_scale=HEAD_DIM**-0.5,
                causal=True,
                window_size=(-1, -1),
                block_table=torch.tensor([[2, 0, 3]], device="cuda", dtype=torch.int32),
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                kv_quant_mode=KVQuantMode.NONE,
            )
            torch.cuda.synchronize()
            reference = _attention_reference(torch, query, keys[:, i], values[:, i])
            errors.append(float((output.float() - reference).abs().max().item()))
        case.update(
            attention_max_abs_error_by_layer=errors, attention_launches=len(errors)
        )
        _validate_case(case, "pre_rope" if pre_rope else "stored_post_rope")
        cases.append(case)
    record = {
        "record_type": RECORD_TYPE,
        "schema_version": 1,
        "scope": QUALIFICATION_SCOPE,
        "bindings": bindings,
        "geometry": _GEOMETRY,
        "runtime": runtime,
        "cases": cases,
        "started_epoch_ns": started,
        "finished_epoch_ns": time.time_ns(),
        "model_forward_executed": False,
        "full_context_semantic_parity_claimed": False,
        "closed_record_sha256": "",
    }
    record["closed_record_sha256"] = _closed_hash(record)
    validated = validate_representative_runtime_qualification_record(
        record, expected_bindings=bindings
    )
    with output_path.open("xb") as stream:
        stream.write(_json_bytes(validated) + b"\n")
    return validate_representative_runtime_qualification_record(
        json.loads(output_path.read_bytes()), expected_bindings=bindings
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    run_representative_runtime_qualification(
        output_path=args.output_json,
        expected_bindings=json.loads(args.bindings_json.read_bytes()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
