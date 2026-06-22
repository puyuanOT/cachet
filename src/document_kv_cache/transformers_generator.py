"""Transformers-backed KV chunk generation for Cachet handoff bundles."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from document_kv_cache.engine_protocol import KVLayout, KVStorageLayout, dtype_byte_width
from document_kv_cache.kvpack import PackChunk
from document_kv_cache.model_profiles import QWEN3_4B_INSTRUCT_HF_MODEL_ID, layout_for_model
from document_kv_cache.models import KVCacheKey
from document_kv_cache.workflow import (
    CacheBuildConfig,
    SourceChunk,
    SourceDocument,
    TrainingArtifacts,
)

CACHET_TRANSFORMERS_MODEL_ID_ENV = "CACHET_TRANSFORMERS_MODEL_ID"
CACHET_TRANSFORMERS_TOKENIZER_ID_ENV = "CACHET_TRANSFORMERS_TOKENIZER_ID"
CACHET_TRANSFORMERS_DEVICE_ENV = "CACHET_TRANSFORMERS_DEVICE"
CACHET_TRANSFORMERS_TORCH_DTYPE_ENV = "CACHET_TRANSFORMERS_TORCH_DTYPE"
CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV = "CACHET_TRANSFORMERS_TRUST_REMOTE_CODE"
CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV = "CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS"
CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV = "CACHET_TRANSFORMERS_CACHE_AXIS_ORDER"
CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV = "CACHET_TRANSFORMERS_MODEL_KWARGS_JSON"
CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV = "CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON"
CACHET_TRANSFORMERS_USE_FAST_TOKENIZER_ENV = "CACHET_TRANSFORMERS_USE_FAST_TOKENIZER"
_CACHE_AXIS_ORDER_HEAD_MAJOR = "head_major"
_CACHE_AXIS_ORDER_TOKEN_MAJOR = "token_major"
_CACHE_AXIS_ORDERS = frozenset(
    {
        _CACHE_AXIS_ORDER_HEAD_MAJOR,
        _CACHE_AXIS_ORDER_TOKEN_MAJOR,
    }
)
_FLOAT_KV_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})
_SUPPORTED_STORAGE_LAYOUTS = frozenset(
    {
        KVStorageLayout.SEPARATE_KEY_VALUE,
        KVStorageLayout.SHARED_KEY_VALUE,
    }
)

__all__ = [
    "CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV",
    "CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV",
    "CACHET_TRANSFORMERS_DEVICE_ENV",
    "CACHET_TRANSFORMERS_MODEL_ID_ENV",
    "CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV",
    "CACHET_TRANSFORMERS_TOKENIZER_ID_ENV",
    "CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV",
    "CACHET_TRANSFORMERS_TORCH_DTYPE_ENV",
    "CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV",
    "CACHET_TRANSFORMERS_USE_FAST_TOKENIZER_ENV",
    "TransformersKVGeneratorConfig",
    "TransformersKVChunkGenerator",
    "build_transformers_kv_chunk_generator",
]


@dataclass(frozen=True, slots=True)
class TransformersKVGeneratorConfig:
    """Configuration for loading a Hugging Face causal LM as a Cachet generator."""

    model_id: str = QWEN3_4B_INSTRUCT_HF_MODEL_ID
    tokenizer_id: str | None = None
    device: str | None = None
    torch_dtype: str | None = "auto"
    trust_remote_code: bool = False
    add_special_tokens: bool = False
    cache_axis_order: str = _CACHE_AXIS_ORDER_HEAD_MAJOR
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _non_empty_string(self.model_id, "model_id"))
        if self.tokenizer_id is not None:
            object.__setattr__(
                self,
                "tokenizer_id",
                _non_empty_string(self.tokenizer_id, "tokenizer_id"),
            )
        if self.device is not None:
            object.__setattr__(self, "device", _non_empty_string(self.device, "device"))
        if self.torch_dtype is not None:
            object.__setattr__(
                self,
                "torch_dtype",
                _non_empty_string(self.torch_dtype, "torch_dtype"),
            )
        if type(self.trust_remote_code) is not bool:
            raise ValueError("trust_remote_code must be a boolean")
        if type(self.add_special_tokens) is not bool:
            raise ValueError("add_special_tokens must be a boolean")
        object.__setattr__(
            self,
            "cache_axis_order",
            _cache_axis_order_from_value(self.cache_axis_order),
        )
        object.__setattr__(
            self,
            "model_kwargs",
            _json_object_mapping(self.model_kwargs, "model_kwargs"),
        )
        object.__setattr__(
            self,
            "tokenizer_kwargs",
            _json_object_mapping(self.tokenizer_kwargs, "tokenizer_kwargs"),
        )

    @property
    def resolved_tokenizer_id(self) -> str:
        return self.model_id if self.tokenizer_id is None else self.tokenizer_id


class TransformersKVChunkGenerator:
    """Generate Cachet ``PackChunk`` payloads from Transformers ``past_key_values``.

    The emitted payload is token-major, then layer-major, then K/V-major. That
    matches Cachet's vLLM native provider, which slices a token span, selects a
    layer, and reshapes the layer values into vLLM-owned paged KV blocks.
    """

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        layout: KVLayout | None = None,
        add_special_tokens: bool = False,
        cache_axis_order: str = _CACHE_AXIS_ORDER_HEAD_MAJOR,
    ) -> None:
        if model is None:
            raise TypeError("model must be provided")
        if tokenizer is None:
            raise TypeError("tokenizer must be provided")
        if type(add_special_tokens) is not bool:
            raise ValueError("add_special_tokens must be a boolean")
        if layout is not None:
            layout.validate()
        self.model = model
        self.tokenizer = tokenizer
        self.layout = layout
        self.add_special_tokens = add_special_tokens
        self.cache_axis_order = _cache_axis_order_from_value(cache_axis_order)

    @classmethod
    def from_pretrained(
        cls,
        config: TransformersKVGeneratorConfig | None = None,
        *,
        layout: KVLayout | None = None,
    ) -> "TransformersKVChunkGenerator":
        """Load a causal LM/tokenizer pair with optional runtime dependencies."""

        resolved = config or TransformersKVGeneratorConfig()
        torch = _torch()
        transformers = _transformers()
        model_kwargs = dict(resolved.model_kwargs)
        if resolved.torch_dtype not in (None, "auto"):
            model_kwargs.setdefault(
                "torch_dtype",
                _torch_dtype_from_name(torch, resolved.torch_dtype),
            )
        elif resolved.torch_dtype == "auto":
            model_kwargs.setdefault("torch_dtype", "auto")
        model_kwargs.setdefault("trust_remote_code", resolved.trust_remote_code)
        tokenizer_kwargs = dict(resolved.tokenizer_kwargs)
        tokenizer_kwargs.setdefault("trust_remote_code", resolved.trust_remote_code)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            resolved.resolved_tokenizer_id,
            **tokenizer_kwargs,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            resolved.model_id,
            **model_kwargs,
        )
        if resolved.device is not None:
            model = model.to(resolved.device)
        evaluator = getattr(model, "eval", None)
        if callable(evaluator):
            evaluator()
        return cls(
            model=model,
            tokenizer=tokenizer,
            layout=layout,
            add_special_tokens=resolved.add_special_tokens,
            cache_axis_order=resolved.cache_axis_order,
        )

    def generate(
        self,
        *,
        document: SourceDocument,
        chunk: SourceChunk,
        config: CacheBuildConfig,
        training_artifacts: TrainingArtifacts | None = None,
    ) -> PackChunk:
        if training_artifacts is not None and training_artifacts.adapter_ids:
            raise ValueError(
                "TransformersKVChunkGenerator does not apply training adapter artifacts"
            )
        layout = _layout_for_config(config, self.layout)
        torch = _torch()
        inputs = self._tokenize(chunk.text)
        input_ids = _input_ids(inputs)
        token_count = int(input_ids.shape[-1])
        if token_count <= 0:
            raise ValueError("tokenized chunk must contain at least one token")
        inputs = _inputs_to_device(inputs, _model_device(self.model))
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
        past_key_values = _past_key_values(outputs)
        payload = _payload_from_past_key_values(
            past_key_values,
            token_count=token_count,
            layout=layout,
            cache_axis_order=self.cache_axis_order,
        )
        expected_bytes = token_count * layout.bytes_per_token
        if len(payload) != expected_bytes:
            raise ValueError(
                f"generated payload length {len(payload)} != expected {expected_bytes}"
            )
        return PackChunk(
            key=KVCacheKey.for_document(
                model_id=config.model_id,
                lora_id=config.lora_id,
                prompt_template_version=config.prompt_template_version,
                document_id=document.document_id,
                chunk_type=chunk.chunk_type,
                chunk_id=chunk.chunk_id,
            ),
            payload=payload,
            token_count=token_count,
            dtype=layout.dtype,
            layout_version=layout.layout_version,
            storage_layout=layout.storage_layout,
        )

    def _tokenize(self, text: str) -> Mapping[str, object]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("tokenizer must return a mapping")
        return encoded


def build_transformers_kv_chunk_generator() -> TransformersKVChunkGenerator:
    """Build a Transformers generator from Cachet environment variables."""

    config = TransformersKVGeneratorConfig(
        model_id=_env_string(CACHET_TRANSFORMERS_MODEL_ID_ENV, default=QWEN3_4B_INSTRUCT_HF_MODEL_ID),
        tokenizer_id=_env_string(CACHET_TRANSFORMERS_TOKENIZER_ID_ENV),
        device=_env_string(CACHET_TRANSFORMERS_DEVICE_ENV),
        torch_dtype=_env_string(CACHET_TRANSFORMERS_TORCH_DTYPE_ENV, default="auto"),
        trust_remote_code=_env_bool(CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV, default=False),
        add_special_tokens=_env_bool(CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV, default=False),
        cache_axis_order=_env_string(
            CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV,
            default=_CACHE_AXIS_ORDER_HEAD_MAJOR,
        ),
        model_kwargs=_env_json_object(CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV),
        tokenizer_kwargs=_tokenizer_kwargs_from_env(),
    )
    return TransformersKVChunkGenerator.from_pretrained(config)


def _tokenizer_kwargs_from_env() -> dict[str, Any]:
    kwargs = _env_json_object(CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV)
    use_fast = _env_optional_bool(CACHET_TRANSFORMERS_USE_FAST_TOKENIZER_ENV)
    if use_fast is not None:
        kwargs["use_fast"] = use_fast
    return kwargs


def _layout_for_config(config: CacheBuildConfig, explicit_layout: KVLayout | None) -> KVLayout:
    if explicit_layout is None:
        layout = layout_for_model(
            config.model_id,
            dtype=config.dtype,
            lora_id=config.lora_id,
            layout_version=config.layout_version,
            storage_layout=config.storage_layout,
        )
    else:
        layout = explicit_layout
    layout.validate()
    expected = {
        "model_id": config.model_id,
        "lora_id": config.lora_id,
        "dtype": config.dtype,
        "layout_version": config.layout_version,
        "storage_layout": config.storage_layout,
    }
    actual = {
        "model_id": layout.model_id,
        "lora_id": layout.lora_id,
        "dtype": layout.dtype,
        "layout_version": layout.layout_version,
        "storage_layout": layout.storage_layout,
    }
    mismatches = tuple(name for name, value in expected.items() if actual[name] != value)
    if mismatches:
        details = ", ".join(
            f"{name}: expected {expected[name]!r}, got {actual[name]!r}"
            for name in mismatches
        )
        raise ValueError(f"layout does not match CacheBuildConfig ({details})")
    if layout.num_kv_heads is None or layout.head_size is None or layout.kv_stride_bytes is None:
        raise ValueError(
            "Transformers KV generation requires num_kv_heads, head_size, and kv_stride_bytes"
        )
    if _dtype_kind(layout.dtype) != "float":
        raise ValueError(
            "Transformers KV generation requires a floating KV dtype such as bfloat16 or float16"
        )
    if layout.storage_layout not in _SUPPORTED_STORAGE_LAYOUTS:
        supported = ", ".join(layout.value for layout in sorted(_SUPPORTED_STORAGE_LAYOUTS))
        raise ValueError(
            f"Transformers KV generation does not support {layout.storage_layout!r}; "
            f"use {supported}"
        )
    return layout


def _payload_from_past_key_values(
    past_key_values: object,
    *,
    token_count: int,
    layout: KVLayout,
    cache_axis_order: str,
) -> bytes:
    torch = _torch()
    legacy_cache = _legacy_past_key_values(past_key_values)
    if len(legacy_cache) != layout.num_layers:
        raise ValueError(
            f"past_key_values layer count {len(legacy_cache)} != layout.num_layers "
            f"{layout.num_layers}"
        )
    dtype = _torch_dtype_from_name(torch, layout.dtype)
    layer_values = []
    for layer_index, layer in enumerate(legacy_cache):
        key, value = _key_value_pair(layer, layer_index=layer_index)
        key_values = _normalize_cache_tensor(
            key,
            token_count=token_count,
            layout=layout,
            label="key",
            cache_axis_order=cache_axis_order,
        )
        value_values = _normalize_cache_tensor(
            value,
            token_count=token_count,
            layout=layout,
            label="value",
            cache_axis_order=cache_axis_order,
        )
        layer_values.append(_layer_values(key_values, value_values, dtype=dtype, layout=layout))
    stacked = torch.stack(layer_values, dim=1).contiguous()
    return _tensor_bytes(stacked)


def _legacy_past_key_values(past_key_values: object) -> tuple[object, ...]:
    converter = getattr(past_key_values, "to_legacy_cache", None)
    if callable(converter):
        past_key_values = converter()
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return tuple(zip(tuple(key_cache), tuple(value_cache), strict=True))
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    raise TypeError("model outputs must include tuple-like past_key_values")


def _key_value_pair(layer: object, *, layer_index: int) -> tuple[object, object]:
    if not isinstance(layer, (tuple, list)) or len(layer) < 2:
        raise TypeError(f"past_key_values[{layer_index}] must contain key and value tensors")
    return layer[0], layer[1]


def _normalize_cache_tensor(
    tensor: object,
    *,
    token_count: int,
    layout: KVLayout,
    label: str,
    cache_axis_order: str,
) -> object:
    torch = _torch()
    if not torch.is_tensor(tensor):
        raise TypeError(f"{label} cache must be a torch.Tensor")
    if tensor.ndim != 4:
        raise ValueError(f"{label} cache must have rank 4")
    if tensor.shape[0] != 1:
        raise ValueError(f"{label} cache batch dimension must be 1")
    assert layout.num_kv_heads is not None
    assert layout.head_size is not None
    cache_axis_order = _cache_axis_order_from_value(cache_axis_order)
    if cache_axis_order == _CACHE_AXIS_ORDER_HEAD_MAJOR:
        if tensor.shape[1] != layout.num_kv_heads or tensor.shape[2] != token_count:
            raise ValueError(f"{label} cache shape must be [1, num_kv_heads, tokens, head_size]")
        normalized = tensor[0].permute(1, 0, 2)
    elif cache_axis_order == _CACHE_AXIS_ORDER_TOKEN_MAJOR:
        if tensor.shape[1] != token_count or tensor.shape[2] != layout.num_kv_heads:
            raise ValueError(f"{label} cache shape must be [1, tokens, num_kv_heads, head_size]")
        normalized = tensor[0]
    else:
        raise AssertionError("unsupported cache axis order")
    if normalized.shape[2] != layout.head_size:
        raise ValueError(f"{label} cache head dimension does not match layout.head_size")
    return normalized.contiguous()


def _layer_values(key: object, value: object, *, dtype: object, layout: KVLayout) -> object:
    torch = _torch()
    key = key.to(dtype=dtype)
    value = value.to(dtype=dtype)
    key = _pad_kv_stride(key, layout=layout)
    value = _pad_kv_stride(value, layout=layout)
    return torch.stack((key, value), dim=1).contiguous()


def _pad_kv_stride(tensor: object, *, layout: KVLayout) -> object:
    torch = _torch()
    assert layout.kv_stride_bytes is not None
    dtype_width = dtype_byte_width(layout.dtype)
    stride_scalars = layout.kv_stride_bytes // dtype_width
    if tensor.shape[-1] == stride_scalars:
        return tensor
    if tensor.shape[-1] > stride_scalars:
        raise ValueError("cache head dimension exceeds layout.kv_stride_bytes")
    padded_shape = (*tuple(tensor.shape[:-1]), stride_scalars)
    padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
    padded[..., : tensor.shape[-1]] = tensor
    return padded


def _tensor_bytes(tensor: object) -> bytes:
    torch = _torch()
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _input_ids(inputs: Mapping[str, object]) -> object:
    input_ids = inputs.get("input_ids")
    torch = _torch()
    if not torch.is_tensor(input_ids):
        raise ValueError("tokenizer output must include tensor input_ids")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("tokenizer input_ids must have shape [1, tokens]")
    return input_ids


def _inputs_to_device(inputs: Mapping[str, object], device: object | None) -> dict[str, object]:
    if device is None:
        return dict(inputs)
    moved = {}
    for key, value in inputs.items():
        mover = getattr(value, "to", None)
        moved[key] = mover(device) if callable(mover) else value
    return moved


def _model_device(model: object) -> object | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except (StopIteration, TypeError):
            return None
    return None


def _past_key_values(outputs: object) -> object:
    if isinstance(outputs, Mapping):
        value = outputs.get("past_key_values")
    else:
        value = getattr(outputs, "past_key_values", None)
    if value is None:
        raise ValueError("model output must include past_key_values")
    return value


def _torch_dtype_from_name(torch: object, dtype: str) -> object:
    normalized = dtype.lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Transformers KV dtype {dtype!r}") from exc


def _dtype_kind(dtype: str) -> str:
    return "float" if dtype.lower() in _FLOAT_KV_DTYPES else "other"


def _cache_axis_order_from_value(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("cache_axis_order must be a non-empty string")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in _CACHE_AXIS_ORDERS:
        supported = ", ".join(sorted(_CACHE_AXIS_ORDERS))
        raise ValueError(f"Unsupported cache_axis_order {value!r}; supported values: {supported}")
    return normalized


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError(
            "Transformers KV generation requires torch in the runtime environment"
        ) from exc
    return torch


def _transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError(
            "Transformers KV generation requires transformers in the runtime environment"
        ) from exc
    return transformers


def _env_string(name: str, *, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value


def _env_bool(name: str, *, default: bool) -> bool:
    value = _env_string(name)
    if value is None:
        return default
    return _bool_from_env_string(name, value)


def _env_optional_bool(name: str) -> bool | None:
    value = _env_string(name)
    if value is None:
        return None
    return _bool_from_env_string(name, value)


def _bool_from_env_string(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value")


def _env_json_object(name: str) -> dict[str, Any]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return {}
    try:
        parsed = _loads_env_json_object(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain a JSON object") from exc
    return _json_object_mapping(parsed, name)


def _loads_env_json_object(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if r"\"" not in value:
            raise
        return json.loads(value.replace(r"\"", '"'))


def _json_object_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        normalized[key] = _json_compatible_value(item, f"{field_name}.{key}")
    return normalized


def _json_compatible_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be JSON-compatible")
        return value
    if isinstance(value, Mapping):
        return _json_object_mapping(value, field_name)
    if isinstance(value, (tuple, list)):
        return [
            _json_compatible_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{field_name} must be JSON-compatible")


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
