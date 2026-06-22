from __future__ import annotations

from types import SimpleNamespace

import pytest

import cachet.transformers_generator as cachet_transformers_generator
import document_kv_cache.transformers_generator as public_transformers_generator
from document_kv_cache.engine_protocol import KVLayout, KVStorageLayout
from document_kv_cache.model_profiles import QWEN3_4B_INSTRUCT_HF_MODEL_ID
from document_kv_cache.transformers_generator import (
    CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV,
    CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV,
    CACHET_TRANSFORMERS_DEVICE_ENV,
    CACHET_TRANSFORMERS_MODEL_ID_ENV,
    CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV,
    CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
    CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
    TransformersKVChunkGenerator,
    TransformersKVGeneratorConfig,
    build_transformers_kv_chunk_generator,
)
from document_kv_cache.workflow import CacheBuildConfig, SourceDocument

torch = pytest.importorskip("torch")


class TinyTokenizer:
    def __init__(self, token_count: int = 2) -> None:
        self.token_count = token_count
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, *, return_tensors: str, add_special_tokens: bool):
        self.calls.append(
            {
                "text": text,
                "return_tensors": return_tensors,
                "add_special_tokens": add_special_tokens,
            }
        )
        return {
            "input_ids": torch.arange(
                self.token_count,
                dtype=torch.long,
            ).reshape(1, self.token_count),
            "attention_mask": torch.ones((1, self.token_count), dtype=torch.long),
        }


class TinyModel:
    def __init__(self, past_key_values) -> None:
        self.past_key_values = past_key_values
        self.calls: list[dict[str, object]] = []

    def __call__(self, *, input_ids, attention_mask=None, use_cache: bool):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": None if attention_mask is None else attention_mask.clone(),
                "use_cache": use_cache,
            }
        )
        return SimpleNamespace(past_key_values=self.past_key_values)


def tiny_layout(
    *,
    dtype: str = "float32",
    num_layers: int = 2,
    num_kv_heads: int = 1,
    num_query_heads: int | None = None,
    head_size: int = 2,
    kv_stride_bytes: int | None = None,
    storage_layout: KVStorageLayout = KVStorageLayout.SEPARATE_KEY_VALUE,
) -> KVLayout:
    dtype_width = 4 if dtype == "float32" else 2 if dtype in {"bfloat16", "float16"} else 1
    stride = head_size * dtype_width if kv_stride_bytes is None else kv_stride_bytes
    shares_kv_storage = storage_layout == KVStorageLayout.SHARED_KEY_VALUE
    return KVLayout(
        model_id="tiny-model",
        lora_id="base",
        layout_version="tiny-v1",
        dtype=dtype,
        num_layers=num_layers,
        block_size=2,
        bytes_per_token=num_layers * num_kv_heads * stride * 2,
        num_query_heads=num_kv_heads if num_query_heads is None else num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        kv_stride_bytes=stride,
        shares_kv_storage=shares_kv_storage,
        storage_layout=storage_layout,
    )


def build_config(layout: KVLayout) -> CacheBuildConfig:
    return CacheBuildConfig(
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version="v1",
        dtype=layout.dtype,
        layout_version=layout.layout_version,
        storage_layout=layout.storage_layout,
    )


def document() -> SourceDocument:
    return SourceDocument.from_text(
        document_id="doc-a",
        text="alpha beta",
        chunk_id="cache_prefix",
    )


def layer(key_values, value_values):
    key = torch.tensor(key_values, dtype=torch.float32).reshape(1, 1, len(key_values), 2)
    value = torch.tensor(value_values, dtype=torch.float32).reshape(1, 1, len(value_values), 2)
    return key, value


def tensor_bytes(tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def test_transformers_generator_emits_token_major_layer_major_payload():
    layout = tiny_layout()
    first_layer = layer([[1, 2], [3, 4]], [[11, 12], [13, 14]])
    second_layer = layer([[21, 22], [23, 24]], [[31, 32], [33, 34]])
    tokenizer = TinyTokenizer(token_count=2)
    model = TinyModel((first_layer, second_layer))
    source = document()
    generator = TransformersKVChunkGenerator(
        model=model,
        tokenizer=tokenizer,
        layout=layout,
        add_special_tokens=True,
    )

    pack_chunk = generator.generate(
        document=source,
        chunk=source.chunks[0],
        config=build_config(layout),
    )

    layer_0 = torch.stack(
        (first_layer[0][0].permute(1, 0, 2), first_layer[1][0].permute(1, 0, 2)),
        dim=1,
    )
    layer_1 = torch.stack(
        (second_layer[0][0].permute(1, 0, 2), second_layer[1][0].permute(1, 0, 2)),
        dim=1,
    )
    expected = torch.stack((layer_0, layer_1), dim=1).contiguous()
    assert pack_chunk.key.document_id == "doc-a"
    assert pack_chunk.key.chunk_id == "cache_prefix"
    assert pack_chunk.token_count == 2
    assert pack_chunk.dtype == "float32"
    assert pack_chunk.layout_version == "tiny-v1"
    assert pack_chunk.storage_layout == KVStorageLayout.SEPARATE_KEY_VALUE
    assert pack_chunk.payload == tensor_bytes(expected)
    assert tokenizer.calls == [
        {
            "text": "alpha beta",
            "return_tensors": "pt",
            "add_special_tokens": True,
        }
    ]
    assert model.calls[0]["use_cache"] is True


def test_transformers_generator_emits_bfloat16_payload_for_shared_layout():
    layout = tiny_layout(dtype="bfloat16", storage_layout=KVStorageLayout.SHARED_KEY_VALUE)
    first_layer = layer([[1, 2]], [[11, 12]])
    second_layer = layer([[21, 22]], [[31, 32]])
    tokenizer = TinyTokenizer(token_count=1)
    model = TinyModel((first_layer, second_layer))
    source = document()
    generator = TransformersKVChunkGenerator(model=model, tokenizer=tokenizer, layout=layout)

    pack_chunk = generator.generate(
        document=source,
        chunk=source.chunks[0],
        config=build_config(layout),
    )

    layer_0 = torch.stack(
        (first_layer[0][0].permute(1, 0, 2), first_layer[1][0].permute(1, 0, 2)),
        dim=1,
    ).to(dtype=torch.bfloat16)
    layer_1 = torch.stack(
        (second_layer[0][0].permute(1, 0, 2), second_layer[1][0].permute(1, 0, 2)),
        dim=1,
    ).to(dtype=torch.bfloat16)
    expected = torch.stack((layer_0, layer_1), dim=1).contiguous()
    assert pack_chunk.dtype == "bfloat16"
    assert pack_chunk.storage_layout == KVStorageLayout.SHARED_KEY_VALUE
    assert len(pack_chunk.payload) == layout.bytes_per_token
    assert pack_chunk.payload == tensor_bytes(expected)


def test_transformers_generator_pads_head_stride_and_accepts_token_major_cache_shape():
    layout = tiny_layout(num_layers=1, kv_stride_bytes=16)
    key = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    value = torch.tensor([[[[11.0, 12.0]], [[13.0, 14.0]]]])
    tokenizer = TinyTokenizer(token_count=2)
    model = TinyModel(((key, value),))
    source = document()
    generator = TransformersKVChunkGenerator(
        model=model,
        tokenizer=tokenizer,
        layout=layout,
        cache_axis_order="token_major",
    )

    pack_chunk = generator.generate(
        document=source,
        chunk=source.chunks[0],
        config=build_config(layout),
    )

    expected = torch.tensor(
        [
            [[[[1.0, 2.0, 0.0, 0.0]], [[11.0, 12.0, 0.0, 0.0]]]],
            [[[[3.0, 4.0, 0.0, 0.0]], [[13.0, 14.0, 0.0, 0.0]]]],
        ]
    )
    assert pack_chunk.token_count == 2
    assert len(pack_chunk.payload) == layout.bytes_per_token * 2
    assert pack_chunk.payload == tensor_bytes(expected)


def test_transformers_generator_token_major_axis_order_handles_ambiguous_shape():
    layout = tiny_layout(num_layers=1, num_kv_heads=2, head_size=1)
    key = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    value = torch.tensor([[[[11.0], [12.0]], [[13.0], [14.0]]]])
    tokenizer = TinyTokenizer(token_count=2)
    model = TinyModel(((key, value),))
    source = document()
    generator = TransformersKVChunkGenerator(
        model=model,
        tokenizer=tokenizer,
        layout=layout,
        cache_axis_order="token_major",
    )

    pack_chunk = generator.generate(
        document=source,
        chunk=source.chunks[0],
        config=build_config(layout),
    )

    expected = torch.stack((key[0], value[0]), dim=1).reshape(2, 1, 2, 2, 1)
    assert pack_chunk.payload == tensor_bytes(expected)


def test_transformers_generator_rejects_integer_payload_dtype():
    layout = tiny_layout(dtype="int8")
    tokenizer = TinyTokenizer(token_count=1)
    model = TinyModel((layer([[1, 2]], [[3, 4]]),))
    source = document()
    generator = TransformersKVChunkGenerator(model=model, tokenizer=tokenizer, layout=layout)

    with pytest.raises(ValueError, match="floating KV dtype"):
        generator.generate(
            document=source,
            chunk=source.chunks[0],
            config=build_config(layout),
        )


def test_transformers_generator_rejects_interleaved_payload_layout():
    layout = tiny_layout(storage_layout=KVStorageLayout.INTERLEAVED_KEY_VALUE)
    tokenizer = TinyTokenizer(token_count=1)
    model = TinyModel((layer([[1, 2]], [[3, 4]]),))
    source = document()
    generator = TransformersKVChunkGenerator(model=model, tokenizer=tokenizer, layout=layout)

    with pytest.raises(ValueError, match="does not support"):
        generator.generate(
            document=source,
            chunk=source.chunks[0],
            config=build_config(layout),
        )


def test_transformers_generator_env_factory_builds_pretrained_config(monkeypatch):
    calls = []
    sentinel = object()

    def fake_from_pretrained(cls, config, *, layout=None):
        calls.append((cls, config, layout))
        return sentinel

    monkeypatch.setattr(
        TransformersKVChunkGenerator,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    monkeypatch.setenv(CACHET_TRANSFORMERS_MODEL_ID_ENV, "model-a")
    monkeypatch.setenv(CACHET_TRANSFORMERS_TOKENIZER_ID_ENV, "tokenizer-a")
    monkeypatch.setenv(CACHET_TRANSFORMERS_DEVICE_ENV, "cuda")
    monkeypatch.setenv(CACHET_TRANSFORMERS_TORCH_DTYPE_ENV, "bfloat16")
    monkeypatch.setenv(CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV, "false")
    monkeypatch.setenv(CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV, "true")
    monkeypatch.setenv(CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV, "token-major")
    monkeypatch.setenv(
        CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV,
        '{"attn_implementation":"eager"}',
    )
    monkeypatch.setenv(CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV, '{"padding_side":"left"}')

    generator = build_transformers_kv_chunk_generator()

    assert generator is sentinel
    assert len(calls) == 1
    _cls, config, layout = calls[0]
    assert layout is None
    assert isinstance(config, TransformersKVGeneratorConfig)
    assert config.model_id == "model-a"
    assert config.tokenizer_id == "tokenizer-a"
    assert config.device == "cuda"
    assert config.torch_dtype == "bfloat16"
    assert config.trust_remote_code is False
    assert config.add_special_tokens is True
    assert config.cache_axis_order == "token_major"
    assert config.model_kwargs == {"attn_implementation": "eager"}
    assert config.tokenizer_kwargs == {"padding_side": "left"}


def test_transformers_generator_env_factory_accepts_databricks_escaped_json(monkeypatch):
    calls = []
    sentinel = object()

    def fake_from_pretrained(cls, config, *, layout=None):
        calls.append((cls, config, layout))
        return sentinel

    monkeypatch.setattr(
        TransformersKVChunkGenerator,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    monkeypatch.setenv(CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV, r"{\"use_fast\":false}")

    generator = build_transformers_kv_chunk_generator()

    assert generator is sentinel
    _cls, config, layout = calls[0]
    assert layout is None
    assert config.tokenizer_kwargs == {"use_fast": False}


def test_transformers_generator_env_factory_treats_blank_values_as_unset(monkeypatch):
    calls = []
    sentinel = object()

    def fake_from_pretrained(cls, config, *, layout=None):
        calls.append((cls, config, layout))
        return sentinel

    monkeypatch.setattr(
        TransformersKVChunkGenerator,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    for name in (
        CACHET_TRANSFORMERS_MODEL_ID_ENV,
        CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
        CACHET_TRANSFORMERS_DEVICE_ENV,
        CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
        CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
        CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV,
        CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV,
        CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV,
        CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV,
    ):
        monkeypatch.setenv(name, " ")

    generator = build_transformers_kv_chunk_generator()

    assert generator is sentinel
    _cls, config, layout = calls[0]
    assert layout is None
    assert config.model_id == QWEN3_4B_INSTRUCT_HF_MODEL_ID
    assert config.tokenizer_id is None
    assert config.device is None
    assert config.torch_dtype == "auto"
    assert config.trust_remote_code is False
    assert config.add_special_tokens is False
    assert config.cache_axis_order == "head_major"
    assert config.model_kwargs == {}
    assert config.tokenizer_kwargs == {}


def test_transformers_generator_public_facade_aliases_document_module():
    assert cachet_transformers_generator is public_transformers_generator
    assert (
        cachet_transformers_generator.TransformersKVChunkGenerator
        is public_transformers_generator.TransformersKVChunkGenerator
    )
