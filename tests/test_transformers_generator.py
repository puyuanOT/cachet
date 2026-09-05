from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

import cachet.transformers_generator as cachet_transformers_generator
import document_kv_cache.transformers_generator as public_transformers_generator
from document_kv_cache.engine_protocol import KVLayout, KVStorageLayout, dtype_byte_width
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.model_profiles import QWEN3_4B_INSTRUCT_HF_MODEL_ID
from document_kv_cache.methods import method_spec
from document_kv_cache.models import CacheGenerationMethod, KVCacheKey
from document_kv_cache.rope import apply_rope_to_keys
from document_kv_cache.transformers_generator import (
    CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV,
    CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV,
    CACHET_TRANSFORMERS_DEVICE_ENV,
    CACHET_TRANSFORMERS_DEVICE_MAP_ENV,
    CACHET_TRANSFORMERS_MODEL_ID_ENV,
    CACHET_TRANSFORMERS_MODEL_REVISION_ENV,
    CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV,
    CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
    CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
    CACHET_TRANSFORMERS_USE_FAST_TOKENIZER_ENV,
    TransformersKVChunkGenerator,
    TransformersKVGeneratorConfig,
    build_post_rope_transformers_kv_chunk_generator,
    build_pre_rope_transformers_kv_chunk_generator,
    build_transformers_kv_chunk_generator,
)
from document_kv_cache.workflow import (
    CacheBuildConfig,
    DocumentKVWorkflow,
    SourceDocument,
)

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


class TinyLogitsToKeepModel(TinyModel):
    def __call__(self, *, input_ids, attention_mask=None, use_cache: bool, logits_to_keep: int):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": None if attention_mask is None else attention_mask.clone(),
                "use_cache": use_cache,
                "logits_to_keep": logits_to_keep,
            }
        )
        return SimpleNamespace(past_key_values=self.past_key_values)


class TinyNumLogitsToKeepModel(TinyModel):
    def __call__(self, *, input_ids, attention_mask=None, use_cache: bool, num_logits_to_keep: int):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": None if attention_mask is None else attention_mask.clone(),
                "use_cache": use_cache,
                "num_logits_to_keep": num_logits_to_keep,
            }
        )
        return SimpleNamespace(past_key_values=self.past_key_values)


class ModernLayerCache:
    def __init__(self, key, value) -> None:
        self.keys = key
        self.values = value


class ModernPastKeyValues:
    def __init__(self, layers) -> None:
        self.layers = [ModernLayerCache(key, value) for key, value in layers]


def tiny_layout(
    *,
    dtype: str = "float32",
    num_layers: int = 2,
    num_kv_heads: int = 1,
    num_query_heads: int | None = None,
    head_size: int = 2,
    kv_stride_bytes: int | None = None,
    storage_layout: KVStorageLayout = KVStorageLayout.SEPARATE_KEY_VALUE,
    payload_axis_order: str = "token_major",
) -> KVLayout:
    dtype_width = dtype_byte_width(dtype)
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
        payload_axis_order=payload_axis_order,
    )


def build_config(
    layout: KVLayout,
    *,
    cache_method: CacheGenerationMethod = CacheGenerationMethod.FULL_PREFIX_PREFILL,
) -> CacheBuildConfig:
    return CacheBuildConfig(
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version="v1",
        dtype=layout.dtype,
        layout_version=layout.layout_version,
        storage_layout=layout.storage_layout,
        payload_axis_order=layout.payload_axis_order,
        cache_method=cache_method,
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
    byte_values = (
        tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
    )
    return bytes(byte_values)


def qwen_like_pre_rope_model(
    monkeypatch,
    *,
    capture_mode: str = "valid",
):
    module = ModuleType(f"cachet_test_qwen_model_{capture_mode}")
    rotary_calls: list[int] = []
    rope_theta = 5_000_000.0
    head_dim = 4

    def roped_key(key):
        positions = torch.arange(key.shape[2], dtype=torch.long)
        normalized = key[0].transpose(0, 1)
        return apply_rope_to_keys(
            normalized,
            positions,
            rope_theta=rope_theta,
            rotary_dim=head_dim,
        ).transpose(0, 1).unsqueeze(0)

    def apply_rotary_pos_emb(query, key, *_args, **_kwargs):
        rotary_calls.append(len(rotary_calls))
        if capture_mode == "bad":
            return query, key + 100.0
        return query, roped_key(key)

    module.apply_rotary_pos_emb = apply_rotary_pos_emb
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class QwenLikeModel:
        config = SimpleNamespace(rope_theta=rope_theta, head_dim=head_dim)

        def __init__(self) -> None:
            self.pre_keys = tuple(
                torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
                + layer_index * 20
                for layer_index in range(2)
            )
            self.values = tuple(
                torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
                + 200
                + layer_index * 20
                for layer_index in range(2)
            )
            self.post_keys: tuple[object, ...] = ()

        def __call__(self, *, input_ids, attention_mask=None, use_cache: bool):
            del input_ids, attention_mask
            assert use_cache is True
            post_keys = []
            for pre_key in self.pre_keys:
                if capture_mode == "missing":
                    post_key = roped_key(pre_key)
                else:
                    _, post_key = module.apply_rotary_pos_emb(
                        torch.zeros_like(pre_key),
                        pre_key,
                    )
                post_keys.append(post_key)
            self.post_keys = tuple(post_keys)
            return SimpleNamespace(
                past_key_values=tuple(zip(self.post_keys, self.values, strict=True))
            )

    QwenLikeModel.__module__ = module.__name__
    model = QwenLikeModel()
    return model, module, apply_rotary_pos_emb, rotary_calls


def test_tensor_bytes_preserves_small_noncontiguous_tensor() -> None:
    tensor = torch.tensor(
        [[1, -2, 300], [-400, 500, -600]],
        dtype=torch.int16,
    ).transpose(0, 1)

    assert public_transformers_generator._tensor_bytes(tensor) == tensor_bytes(tensor)


def test_tensor_bytes_uses_py_ssize_t_above_signed_ctypes_size(
    monkeypatch,
) -> None:
    byte_count = (1 << 31) + 17
    data_ptr = 0x1_0000_0000

    class FakeLargeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def view(self, dtype):
            assert dtype is torch.uint8
            return self

        def numel(self):
            return byte_count

        def data_ptr(self):
            return data_ptr

    class FakeBytesFromPointer:
        def __init__(self) -> None:
            self.argtypes = None
            self.restype = None
            self.calls = []

        def __call__(self, pointer, size):
            self.calls.append((pointer, size))
            return b"large-payload"

    bytes_from_pointer = FakeBytesFromPointer()
    pythonapi = SimpleNamespace(PyBytes_FromStringAndSize=bytes_from_pointer)
    monkeypatch.setattr(public_transformers_generator.ctypes, "pythonapi", pythonapi)

    assert (
        public_transformers_generator._tensor_bytes(FakeLargeTensor())
        == b"large-payload"
    )
    assert bytes_from_pointer.argtypes == (
        public_transformers_generator.ctypes.c_void_p,
        public_transformers_generator.ctypes.c_ssize_t,
    )
    assert bytes_from_pointer.restype is public_transformers_generator.ctypes.py_object
    assert len(bytes_from_pointer.calls) == 1
    pointer, size = bytes_from_pointer.calls[0]
    assert isinstance(pointer, public_transformers_generator.ctypes.c_void_p)
    assert pointer.value == data_ptr
    assert isinstance(size, public_transformers_generator.ctypes.c_ssize_t)
    assert size.value == byte_count


def test_tensor_bytes_fails_closed_without_cpython(monkeypatch) -> None:
    fake_sys = SimpleNamespace(
        implementation=SimpleNamespace(name="pypy"),
        maxsize=public_transformers_generator.sys.maxsize,
    )
    monkeypatch.setattr(public_transformers_generator, "sys", fake_sys)

    with pytest.raises(RuntimeError, match="requires the CPython C API"):
        public_transformers_generator._tensor_bytes(torch.tensor([1], dtype=torch.uint8))


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
    assert pack_chunk.key.content_hash == hashlib.sha256(pack_chunk.payload).hexdigest()
    assert tokenizer.calls == [
        {
            "text": "alpha beta",
            "return_tensors": "pt",
            "add_special_tokens": True,
        }
    ]
    assert model.calls[0]["use_cache"] is True


def test_pre_rope_generator_captures_keys_preserves_values_and_restores_hook(
    monkeypatch,
):
    model, module, original_hook, rotary_calls = qwen_like_pre_rope_model(
        monkeypatch
    )
    layout = replace(
        tiny_layout(num_layers=2, head_size=4),
        pre_rope=True,
        rope_theta=5_000_000.0,
        rope_rotary_dim=4,
        key_position_encoding="pre_rope",
    )
    generator = TransformersKVChunkGenerator(
        model=model,
        tokenizer=TinyTokenizer(token_count=2),
        layout=layout,
        pre_rope=True,
    )
    source = document()

    pack_chunk = generator.generate(
        document=source,
        chunk=source.chunks[0],
        config=build_config(
            layout,
            cache_method=CacheGenerationMethod.VANILLA_PREFILL,
        ),
    )

    expected_pre_rope = public_transformers_generator._payload_from_past_key_values(
        tuple(zip(model.pre_keys, model.values, strict=True)),
        token_count=2,
        layout=layout,
        cache_axis_order="head_major",
    )
    post_rope = public_transformers_generator._payload_from_past_key_values(
        tuple(zip(model.post_keys, model.values, strict=True)),
        token_count=2,
        layout=layout,
        cache_axis_order="head_major",
    )
    assert pack_chunk.payload == expected_pre_rope
    assert pack_chunk.payload != post_rope
    assert rotary_calls == [0, 1]
    assert module.apply_rotary_pos_emb is original_hook
    assert pack_chunk.key.artifact_identity is not None
    assert pack_chunk.key.artifact_identity.method_version == "2"
    assert pack_chunk.key.artifact_identity.key_position_encoding == "pre_rope"
    assert pack_chunk.key.artifact_identity.rope_theta == 5_000_000.0
    assert pack_chunk.key.artifact_identity.rope_rotary_dim == 4


def test_pre_rope_layout_binding_is_idempotent_and_rejects_real_conflicts() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(rope_theta=5_000_000.0, head_dim=4)
    )
    generator = TransformersKVChunkGenerator(
        model=model,
        tokenizer=TinyTokenizer(),
        pre_rope=True,
    )
    layout = replace(
        tiny_layout(num_layers=2, head_size=4),
        pre_rope=True,
        rope_theta=5_000_000.0,
        rope_rotary_dim=4,
        key_position_encoding="pre_rope",
        shares_kv_storage=False,
        storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE,
    )

    generator.bind_layout(layout)
    generator.bind_layout(layout)
    assert generator.layout == layout

    with pytest.raises(
        ValueError,
        match="generator layout conflicts with the resolved handoff layout",
    ):
        generator.bind_layout(replace(layout, lora_id="conflicting-adapter"))


@pytest.mark.parametrize(
    ("capture_mode", "message"),
    (
        ("missing", "captured pre-RoPE key count 0 != layer count 2"),
        ("bad", "pre-RoPE self-check failed"),
    ),
)
def test_pre_rope_generator_rejects_missing_or_bad_capture_and_restores_hook(
    monkeypatch,
    capture_mode,
    message,
):
    model, module, original_hook, _rotary_calls = qwen_like_pre_rope_model(
        monkeypatch,
        capture_mode=capture_mode,
    )
    layout = replace(
        tiny_layout(num_layers=2, head_size=4),
        pre_rope=True,
        rope_theta=5_000_000.0,
        rope_rotary_dim=4,
        key_position_encoding="pre_rope",
    )
    generator = TransformersKVChunkGenerator(
        model=model,
        tokenizer=TinyTokenizer(token_count=2),
        layout=layout,
        pre_rope=True,
    )
    source = document()

    with pytest.raises(ValueError, match=message):
        generator.generate(
            document=source,
            chunk=source.chunks[0],
            config=build_config(
                layout,
                cache_method=CacheGenerationMethod.VANILLA_PREFILL,
            ),
        )

    assert module.apply_rotary_pos_emb is original_hook


def test_registered_vanilla_generator_derives_pre_rope_layout_in_workflow(
    tmp_path,
    monkeypatch,
):
    model, _module, _original_hook, _rotary_calls = qwen_like_pre_rope_model(
        monkeypatch
    )
    resolved_layout = replace(
        tiny_layout(num_layers=2, head_size=4),
        pre_rope=True,
        rope_theta=5_000_000.0,
        rope_rotary_dim=4,
        key_position_encoding="pre_rope",
    )
    generator = TransformersKVChunkGenerator(
        model=model,
        tokenizer=TinyTokenizer(token_count=2),
        layout=None,
        pre_rope=True,
    )
    layout_calls = []

    def fake_layout_for_model(model_id, **kwargs):
        layout_calls.append((model_id, kwargs))
        return resolved_layout

    monkeypatch.setattr(
        public_transformers_generator,
        "build_pre_rope_transformers_kv_chunk_generator",
        lambda: generator,
    )
    monkeypatch.setattr(
        public_transformers_generator,
        "layout_for_model",
        fake_layout_for_model,
    )
    vanilla = method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    created = vanilla.create_generator()
    config = build_config(
        resolved_layout,
        cache_method=CacheGenerationMethod.VANILLA_PREFILL,
    )
    workflow = DocumentKVWorkflow.with_storage(manifest=InMemoryManifestStore())

    result = workflow.generate_cache(
        documents=(document(),),
        generator=created,
        config=config,
        shard_uri=tmp_path / "vanilla.kvpack",
        align_bytes=1,
    )

    assert result.artifact_identity is not None
    assert result.artifact_identity.method_id == "vanilla_prefill"
    assert result.artifact_identity.method_version == "2"
    assert result.artifact_identity.key_position_encoding == "pre_rope"
    assert result.artifact_identity.rope_theta == 5_000_000.0
    assert result.artifact_identity.rope_rotary_dim == 4
    assert layout_calls == [
        (
            resolved_layout.model_id,
            {
                "dtype": resolved_layout.dtype,
                "lora_id": resolved_layout.lora_id,
                "layout_version": resolved_layout.layout_version,
                "storage_layout": resolved_layout.storage_layout,
                "payload_axis_order": resolved_layout.payload_axis_order,
                "pre_rope": True,
                "rope_theta": 5_000_000.0,
                "rope_rotary_dim": 4,
                "shares_kv_storage": False,
            },
        )
    ]


@pytest.mark.parametrize(
    "cache_method",
    (
        CacheGenerationMethod.VANILLA_PREFILL,
        CacheGenerationMethod.FULL_PREFIX_PREFILL,
    ),
)
def test_strict_workflow_rejects_method_position_mislabelling_before_write(
    tmp_path,
    monkeypatch,
    cache_method,
):
    if cache_method == CacheGenerationMethod.VANILLA_PREFILL:
        model, _module, _original_hook, _rotary_calls = qwen_like_pre_rope_model(
            monkeypatch
        )
        layout = replace(
            tiny_layout(num_layers=2, head_size=4),
            pre_rope=True,
            rope_theta=5_000_000.0,
            rope_rotary_dim=4,
            key_position_encoding="pre_rope",
        )
        delegate = TransformersKVChunkGenerator(
            model=model,
            tokenizer=TinyTokenizer(token_count=2),
            layout=layout,
            pre_rope=True,
        )
        claimed_encoding = "stored_post_rope"
    else:
        layout = tiny_layout(num_layers=2)
        delegate = TransformersKVChunkGenerator(
            model=TinyModel(
                (
                    layer([[1, 2], [3, 4]], [[11, 12], [13, 14]]),
                    layer([[21, 22], [23, 24]], [[31, 32], [33, 34]]),
                )
            ),
            tokenizer=TinyTokenizer(token_count=2),
            layout=layout,
        )
        claimed_encoding = "pre_rope"

    class MislabelledGenerator:
        pre_rope = delegate.pre_rope
        position_handling = delegate.position_handling

        def generate(self, **kwargs):
            pack_chunk = delegate.generate(**kwargs)
            identity = pack_chunk.key.artifact_identity
            assert identity is not None
            if claimed_encoding == "pre_rope":
                identity = replace(
                    identity,
                    key_position_encoding=claimed_encoding,
                    rope_theta=5_000_000.0,
                    rope_rotary_dim=layout.head_size,
                )
            else:
                identity = replace(
                    identity,
                    key_position_encoding=claimed_encoding,
                    rope_theta=None,
                    rope_rotary_dim=None,
                )
            return replace(
                pack_chunk,
                key=KVCacheKey.for_document(
                    model_id=pack_chunk.key.model_id,
                    lora_id=pack_chunk.key.lora_id,
                    prompt_template_version=pack_chunk.key.prompt_template_version,
                    document_id=pack_chunk.key.document_id,
                    chunk_type=pack_chunk.key.chunk_type,
                    chunk_id=pack_chunk.key.chunk_id,
                    content_hash=pack_chunk.key.content_hash,
                    artifact_identity=identity,
                    token_contract=pack_chunk.key.token_contract,
                ),
            )

    manifest = InMemoryManifestStore()
    workflow = DocumentKVWorkflow.with_storage(manifest=manifest)
    shard_path = tmp_path / f"mislabelled-{cache_method.value}.kvpack"

    with pytest.raises(ValueError, match="position encoding does not match"):
        workflow.generate_cache(
            documents=(document(),),
            generator=MislabelledGenerator(),
            config=build_config(layout, cache_method=cache_method),
            shard_uri=shard_path,
            align_bytes=1,
        )

    assert not shard_path.exists()
    assert manifest.keys_for_document("doc-a") == []


def test_transformers_generator_emits_layer_major_payload():
    layout = tiny_layout(payload_axis_order="layer_major")
    first_layer = layer([[1, 2], [3, 4]], [[11, 12], [13, 14]])
    second_layer = layer([[21, 22], [23, 24]], [[31, 32], [33, 34]])
    tokenizer = TinyTokenizer(token_count=2)
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
    )
    layer_1 = torch.stack(
        (second_layer[0][0].permute(1, 0, 2), second_layer[1][0].permute(1, 0, 2)),
        dim=1,
    )
    # layer-major stacks the per-layer tensors on a new leading axis so each
    # layer's whole token span is one contiguous block: [layer, token, K/V, ...].
    expected = torch.stack((layer_0, layer_1), dim=0).contiguous()
    token_major_expected = torch.stack((layer_0, layer_1), dim=1).contiguous()

    assert pack_chunk.payload == tensor_bytes(expected)
    assert pack_chunk.payload != tensor_bytes(token_major_expected)
    assert len(pack_chunk.payload) == layout.bytes_per_token * 2
    # Layer 0's token span is the contiguous prefix of the layer-major payload.
    assert pack_chunk.payload[: len(tensor_bytes(layer_0))] == tensor_bytes(layer_0)


def test_transformers_generator_accepts_transformers5_layer_cache():
    layout = tiny_layout()
    first_layer = layer([[1, 2], [3, 4]], [[11, 12], [13, 14]])
    second_layer = layer([[21, 22], [23, 24]], [[31, 32], [33, 34]])
    tokenizer = TinyTokenizer(token_count=2)
    model = TinyModel(ModernPastKeyValues((first_layer, second_layer)))
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
    )
    layer_1 = torch.stack(
        (second_layer[0][0].permute(1, 0, 2), second_layer[1][0].permute(1, 0, 2)),
        dim=1,
    )
    expected = torch.stack((layer_0, layer_1), dim=1).contiguous()
    assert pack_chunk.payload == tensor_bytes(expected)


def test_transformers_generator_requests_single_logits_position_when_supported():
    layout = tiny_layout(num_layers=1)
    first_layer = layer([[1, 2], [3, 4]], [[11, 12], [13, 14]])
    tokenizer = TinyTokenizer(token_count=2)
    model = TinyLogitsToKeepModel((first_layer,))
    source = document()
    generator = TransformersKVChunkGenerator(model=model, tokenizer=tokenizer, layout=layout)

    pack_chunk = generator.generate(
        document=source,
        chunk=source.chunks[0],
        config=build_config(layout),
    )

    expected = torch.stack(
        (first_layer[0][0].permute(1, 0, 2), first_layer[1][0].permute(1, 0, 2)),
        dim=1,
    ).unsqueeze(1)
    assert pack_chunk.payload == tensor_bytes(expected)
    assert model.calls[0]["logits_to_keep"] == 1


def test_transformers_generator_supports_num_logits_to_keep_fallback():
    layout = tiny_layout(num_layers=1)
    first_layer = layer([[1, 2]], [[11, 12]])
    tokenizer = TinyTokenizer(token_count=1)
    model = TinyNumLogitsToKeepModel((first_layer,))
    source = document()
    generator = TransformersKVChunkGenerator(model=model, tokenizer=tokenizer, layout=layout)

    pack_chunk = generator.generate(
        document=source,
        chunk=source.chunks[0],
        config=build_config(layout),
    )

    expected = torch.stack(
        (first_layer[0][0].permute(1, 0, 2), first_layer[1][0].permute(1, 0, 2)),
        dim=1,
    ).unsqueeze(1)
    assert pack_chunk.payload == tensor_bytes(expected)
    assert model.calls[0]["num_logits_to_keep"] == 1


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

    with pytest.raises(ValueError, match="floating KV dtype or FP8"):
        generator.generate(
            document=source,
            chunk=source.chunks[0],
            config=build_config(layout),
        )


def test_transformers_generator_can_emit_fp8_payload():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch runtime does not expose float8_e4m3fn")
    layout = tiny_layout(dtype="fp8")
    first_layer = layer([[1, 2], [3, 4]], [[11, 12], [13, 14]])
    second_layer = layer([[21, 22], [23, 24]], [[31, 32], [33, 34]])
    tokenizer = TinyTokenizer(token_count=2)
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
    ).to(dtype=torch.float8_e4m3fn).view(torch.uint8)
    layer_1 = torch.stack(
        (second_layer[0][0].permute(1, 0, 2), second_layer[1][0].permute(1, 0, 2)),
        dim=1,
    ).to(dtype=torch.float8_e4m3fn).view(torch.uint8)
    expected = torch.stack((layer_0, layer_1), dim=1).contiguous()
    assert pack_chunk.dtype == "fp8"
    assert len(pack_chunk.payload) == layout.bytes_per_token * 2
    assert pack_chunk.payload == tensor_bytes(expected)


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


def test_transformers_generator_from_pretrained_configures_bitsandbytes_quantization(monkeypatch):
    calls = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["tokenizer"] = (model_id, kwargs)
            return TinyTokenizer()

    class FakeModel:
        def __init__(self) -> None:
            self.to_calls = []
            self.eval_called = False

        def to(self, device):
            self.to_calls.append(device)
            return self

        def eval(self):
            self.eval_called = True
            return self

    fake_model = FakeModel()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return fake_model

    fake_transformers = SimpleNamespace(
        __version__="5.12.1",
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    monkeypatch.setattr(public_transformers_generator, "_transformers", lambda: fake_transformers)

    generator = TransformersKVChunkGenerator.from_pretrained(
        TransformersKVGeneratorConfig(
            model_id="model-a",
            model_revision="model-revision-a",
            tokenizer_id="tokenizer-a",
            tokenizer_revision="tokenizer-revision-a",
            device="cuda",
            torch_dtype="bfloat16",
            quantization="bitsandbytes-4bit",
            quantization_config={
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_storage": "uint8",
            },
        )
    )

    assert isinstance(generator, TransformersKVChunkGenerator)
    assert calls["tokenizer"] == (
        "tokenizer-a",
        {"revision": "tokenizer-revision-a", "trust_remote_code": False},
    )
    model_id, model_kwargs = calls["model"]
    assert model_id == "model-a"
    assert model_kwargs["torch_dtype"] is torch.bfloat16
    assert model_kwargs["device_map"] == "cuda"
    assert model_kwargs["trust_remote_code"] is False
    assert model_kwargs["revision"] == "model-revision-a"
    assert model_kwargs["quantization_config"].kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_quant_storage": torch.uint8,
    }
    assert fake_model.to_calls == []
    assert fake_model.eval_called is True
    assert generator.model_id == "model-a"
    assert generator.model_revision == "model-revision-a"
    assert generator.tokenizer_id == "tokenizer-a"
    assert generator.tokenizer_revision == "tokenizer-revision-a"
    assert generator.generator_version == "5.12.1"


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        (
            {
                "model_revision": "pinned-model",
                "model_kwargs": {"revision": "other-model"},
            },
            "model_kwargs.revision conflicts",
        ),
        (
            {
                "tokenizer_revision": "pinned-tokenizer",
                "tokenizer_kwargs": {"revision": "other-tokenizer"},
            },
            "tokenizer_kwargs.revision conflicts",
        ),
    ],
)
def test_transformers_generator_rejects_conflicting_pinned_revision(
    monkeypatch,
    config_kwargs,
    message,
):
    monkeypatch.setattr(
        public_transformers_generator,
        "_transformers",
        lambda: SimpleNamespace(__version__="5.12.1"),
    )

    with pytest.raises(ValueError, match=message):
        TransformersKVChunkGenerator.from_pretrained(
            TransformersKVGeneratorConfig(**config_kwargs)
        )


def test_transformers_generator_env_factory_builds_pretrained_config(monkeypatch):
    calls = []
    sentinel = object()

    def fake_from_pretrained(cls, config, *, layout=None, pre_rope=False):
        calls.append((cls, config, layout))
        return sentinel

    monkeypatch.setattr(
        TransformersKVChunkGenerator,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    monkeypatch.setenv(CACHET_TRANSFORMERS_MODEL_ID_ENV, "model-a")
    monkeypatch.setenv(CACHET_TRANSFORMERS_MODEL_REVISION_ENV, "model-revision-a")
    monkeypatch.setenv(CACHET_TRANSFORMERS_TOKENIZER_ID_ENV, "tokenizer-a")
    monkeypatch.setenv(
        CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
        "tokenizer-revision-a",
    )
    monkeypatch.setenv(CACHET_TRANSFORMERS_DEVICE_ENV, "cuda")
    monkeypatch.setenv(CACHET_TRANSFORMERS_TORCH_DTYPE_ENV, "bfloat16")
    monkeypatch.setenv(CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV, "false")
    monkeypatch.setenv(CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV, "true")
    monkeypatch.setenv(CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV, "token-major")
    monkeypatch.setenv(CACHET_TRANSFORMERS_QUANTIZATION_ENV, "bitsandbytes-4bit")
    monkeypatch.setenv(
        CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
        '{"bnb_4bit_compute_dtype":"bfloat16"}',
    )
    monkeypatch.setenv(CACHET_TRANSFORMERS_DEVICE_MAP_ENV, "auto")
    monkeypatch.setenv(
        CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV,
        '{"attn_implementation":"eager"}',
    )
    monkeypatch.setenv(CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV, '{"padding_side":"left"}')
    monkeypatch.setenv(CACHET_TRANSFORMERS_USE_FAST_TOKENIZER_ENV, "false")

    generator = build_transformers_kv_chunk_generator()

    assert generator is sentinel
    assert len(calls) == 1
    _cls, config, layout = calls[0]
    assert layout is None
    assert isinstance(config, TransformersKVGeneratorConfig)
    assert config.model_id == "model-a"
    assert config.model_revision == "model-revision-a"
    assert config.tokenizer_id == "tokenizer-a"
    assert config.tokenizer_revision == "tokenizer-revision-a"
    assert config.device == "cuda"
    assert config.torch_dtype == "bfloat16"
    assert config.trust_remote_code is False
    assert config.add_special_tokens is True
    assert config.cache_axis_order == "token_major"
    assert config.quantization == "bitsandbytes-4bit"
    assert config.quantization_config == {"bnb_4bit_compute_dtype": "bfloat16"}
    assert config.device_map == "auto"
    assert config.model_kwargs == {"attn_implementation": "eager"}
    assert config.tokenizer_kwargs == {"padding_side": "left", "use_fast": False}


@pytest.mark.parametrize(
    ("factory", "expected_pre_rope"),
    (
        (build_post_rope_transformers_kv_chunk_generator, False),
        (build_pre_rope_transformers_kv_chunk_generator, True),
    ),
)
def test_contract_factories_force_position_encoding(
    monkeypatch,
    factory,
    expected_pre_rope,
):
    observed = []
    sentinel = object()

    def fake_from_pretrained(cls, config, *, layout=None, pre_rope=False):
        observed.append(pre_rope)
        return sentinel

    monkeypatch.setattr(
        TransformersKVChunkGenerator,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    monkeypatch.setenv(
        public_transformers_generator.CACHET_TRANSFORMERS_PRE_ROPE_ENV,
        str(not expected_pre_rope),
    )

    assert factory() is sentinel
    assert observed == [expected_pre_rope]


def test_transformers_generator_env_factory_accepts_databricks_escaped_json(monkeypatch):
    calls = []
    sentinel = object()

    def fake_from_pretrained(cls, config, *, layout=None, pre_rope=False):
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


def test_transformers_generator_env_factory_use_fast_env_overrides_json(monkeypatch):
    calls = []
    sentinel = object()

    def fake_from_pretrained(cls, config, *, layout=None, pre_rope=False):
        calls.append((cls, config, layout))
        return sentinel

    monkeypatch.setattr(
        TransformersKVChunkGenerator,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    monkeypatch.setenv(CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV, '{"use_fast":true}')
    monkeypatch.setenv(CACHET_TRANSFORMERS_USE_FAST_TOKENIZER_ENV, "false")

    generator = build_transformers_kv_chunk_generator()

    assert generator is sentinel
    _cls, config, layout = calls[0]
    assert layout is None
    assert config.tokenizer_kwargs == {"use_fast": False}


def test_transformers_generator_env_factory_treats_blank_values_as_unset(monkeypatch):
    calls = []
    sentinel = object()

    def fake_from_pretrained(cls, config, *, layout=None, pre_rope=False):
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
        CACHET_TRANSFORMERS_QUANTIZATION_ENV,
        CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
        CACHET_TRANSFORMERS_DEVICE_MAP_ENV,
        CACHET_TRANSFORMERS_MODEL_KWARGS_JSON_ENV,
        CACHET_TRANSFORMERS_TOKENIZER_KWARGS_JSON_ENV,
        CACHET_TRANSFORMERS_USE_FAST_TOKENIZER_ENV,
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
    assert config.quantization is None
    assert config.quantization_config == {}
    assert config.device_map is None
    assert config.model_kwargs == {}
    assert config.tokenizer_kwargs == {}


def test_transformers_generator_public_facade_aliases_document_module():
    assert cachet_transformers_generator is public_transformers_generator
    assert (
        cachet_transformers_generator.TransformersKVChunkGenerator
        is public_transformers_generator.TransformersKVChunkGenerator
    )
