import pytest

from document_kv_cache.engine_protocol import AttentionMechanism, KVStorageLayout
from document_kv_cache.model_profiles import (
    KVModelProfile,
    MODEL_PROFILE_RECORD_TYPE,
    ModelProfileDefinition,
    ModelProfileRegistry,
    QWEN3_4B_BASE_HF_MODEL_ID,
    QWEN3_4B_INSTRUCT_HF_MODEL_ID,
    QWEN3_4B_INSTRUCT_PROFILE,
    builtin_model_profiles,
    default_model_profile_registry,
    dtype_byte_width,
    get_model_profile,
    layout_for_model,
    model_profile_definition_from_record,
    model_profile_definition_to_record,
    read_model_profile_definition_json,
    write_model_profile_definition_json,
)


def test_qwen3_builtin_profile_derives_gqa_layout():
    profile = QWEN3_4B_INSTRUCT_PROFILE

    assert profile.metadata["hf_model_id"] == QWEN3_4B_INSTRUCT_HF_MODEL_ID
    assert QWEN3_4B_INSTRUCT_HF_MODEL_ID == "Qwen/Qwen3-4B-Instruct-2507"
    assert QWEN3_4B_BASE_HF_MODEL_ID == "Qwen/Qwen3-4B"
    assert profile.max_context_tokens == 262144
    assert profile.attention_mechanism == AttentionMechanism.GROUPED_QUERY
    assert profile.query_heads_per_kv_head == 4
    assert profile.kv_scalars_per_token == 36 * 8 * 128 * 2
    assert profile.bytes_per_token("int8") == 73728
    assert profile.bytes_per_token("bf16") == 147456

    layout = profile.to_layout(lora_id="selection-lora", dtype="int8")

    assert layout.model_id == "qwen3:4b-instruct"
    assert layout.lora_id == "selection-lora"
    assert layout.num_layers == 36
    assert layout.num_query_heads == 32
    assert layout.num_kv_heads == 8
    assert layout.head_size == 128
    assert layout.kv_stride_bytes == 128
    assert layout.bytes_per_token == 73728
    assert layout.expected_bytes_per_token == 73728
    assert layout.attention_mechanism == AttentionMechanism.GROUPED_QUERY
    assert layout.query_heads_per_kv_head == 4
    assert layout.shares_kv_storage is True
    assert layout.storage_layout == KVStorageLayout.SHARED_KEY_VALUE


def test_profile_metadata_is_immutable():
    profile = QWEN3_4B_INSTRUCT_PROFILE

    with pytest.raises(TypeError):
        profile.metadata["attention"] = "mha"  # type: ignore[index]


def test_builtin_profile_aliases_and_layout_helper():
    profiles = builtin_model_profiles()

    assert profiles["qwen3:4b-instruct"] is QWEN3_4B_INSTRUCT_PROFILE
    assert profiles[QWEN3_4B_INSTRUCT_HF_MODEL_ID] is QWEN3_4B_INSTRUCT_PROFILE
    assert profiles[QWEN3_4B_BASE_HF_MODEL_ID] is QWEN3_4B_INSTRUCT_PROFILE
    assert default_model_profile_registry().get("qwen3:4b-instruct") is QWEN3_4B_INSTRUCT_PROFILE
    assert get_model_profile(QWEN3_4B_INSTRUCT_HF_MODEL_ID) is QWEN3_4B_INSTRUCT_PROFILE
    assert get_model_profile(QWEN3_4B_BASE_HF_MODEL_ID) is QWEN3_4B_INSTRUCT_PROFILE
    alias_layout = layout_for_model(QWEN3_4B_INSTRUCT_HF_MODEL_ID, dtype="bf16")
    assert alias_layout.model_id == QWEN3_4B_INSTRUCT_HF_MODEL_ID
    assert alias_layout.bytes_per_token == 147456
    assert alias_layout.storage_layout == KVStorageLayout.SHARED_KEY_VALUE


def test_custom_model_profile_registry_extends_aliases_without_mutating_builtins():
    future_profile = KVModelProfile(
        model_id="minimax:m2.5-4b",
        architecture="MiniMaxForCausalLM",
        num_layers=28,
        num_query_heads=32,
        num_kv_heads=4,
        head_size=128,
        max_context_tokens=65536,
        default_layout_version="minimax-m2.5-v1",
        metadata={"status": "future-extension"},
    )
    base_registry = default_model_profile_registry()

    extended_registry = base_registry.with_profile(
        future_profile,
        aliases=("MiniMaxAI/MiniMax-M2.5-4B", "qwen3.5:4b-instruct"),
    )

    assert "MiniMaxAI/MiniMax-M2.5-4B" not in base_registry
    assert "qwen3.5:4b-instruct" not in builtin_model_profiles()
    assert extended_registry.get("MiniMaxAI/MiniMax-M2.5-4B") is future_profile
    assert extended_registry.get("qwen3.5:4b-instruct") is future_profile

    layout = extended_registry.layout_for_model("MiniMaxAI/MiniMax-M2.5-4B", dtype="bf16", lora_id="future-lora")

    assert layout.model_id == "MiniMaxAI/MiniMax-M2.5-4B"
    assert layout.lora_id == "future-lora"
    assert layout.attention_mechanism == AttentionMechanism.GROUPED_QUERY
    assert layout.query_heads_per_kv_head == 8
    assert layout.bytes_per_token == future_profile.bytes_per_token("bf16")


def test_model_profile_layout_derives_bytes_from_padded_kv_stride():
    layout = QWEN3_4B_INSTRUCT_PROFILE.to_layout(dtype="bf16", kv_stride_bytes=320)

    assert layout.kv_stride_bytes == 320
    assert layout.bytes_per_token == 36 * 8 * 320 * 2
    assert layout.expected_bytes_per_token == 36 * 8 * 320 * 2
    assert (
        QWEN3_4B_INSTRUCT_PROFILE.bytes_per_token("bf16", kv_stride_bytes=320)
        == 36 * 8 * 320 * 2
    )


def test_model_profile_registry_accepts_generator_aliases_and_rejects_collisions():
    custom_profile = KVModelProfile(
        model_id="custom:4b",
        architecture="CustomForCausalLM",
        num_layers=24,
        num_query_heads=32,
        num_kv_heads=4,
        head_size=128,
        max_context_tokens=65536,
        default_layout_version="custom-v1",
    )
    conflicting_profile = KVModelProfile(
        model_id="qwen3:4b-instruct",
        architecture="ConflictingQwenForCausalLM",
        num_layers=24,
        num_query_heads=32,
        num_kv_heads=4,
        head_size=128,
        max_context_tokens=65536,
        default_layout_version="custom-qwen-v1",
    )
    registry = default_model_profile_registry()

    extended = registry.with_profile(custom_profile, aliases=(alias for alias in ("Custom/4B",)))

    assert extended.get("Custom/4B") is custom_profile
    assert "Custom/4B" not in registry
    with pytest.raises(ValueError, match="already registered"):
        registry.with_profile(custom_profile, aliases=(QWEN3_4B_INSTRUCT_HF_MODEL_ID,))
    with pytest.raises(ValueError, match="already registered"):
        registry.with_profile(conflicting_profile)


def test_model_profile_definition_round_trips_future_model_artifact(tmp_path):
    future_profile = KVModelProfile(
        model_id="minimax:m2.5-4b",
        architecture="MiniMaxForCausalLM",
        num_layers=28,
        num_query_heads=32,
        num_kv_heads=1,
        head_size=128,
        max_context_tokens=65536,
        default_layout_version="minimax-m2.5-v1",
        default_dtype="int8",
        default_block_size=32,
        default_lora_id="base",
        metadata={"status": "future-extension", "attention": "mqa"},
    )
    definition = ModelProfileDefinition(
        profile=future_profile,
        aliases=("MiniMaxAI/MiniMax-M2.5-4B", "qwen3.5:4b-instruct"),
    )

    record = model_profile_definition_to_record(definition)
    round_tripped = model_profile_definition_from_record(record)
    output_path = tmp_path / "profiles" / "minimax.json"
    write_model_profile_definition_json(definition, output_path)
    from_file = read_model_profile_definition_json(output_path)

    assert record == {
        "record_type": MODEL_PROFILE_RECORD_TYPE,
        "model_id": "minimax:m2.5-4b",
        "architecture": "MiniMaxForCausalLM",
        "num_layers": 28,
        "num_query_heads": 32,
        "num_kv_heads": 1,
        "head_size": 128,
        "max_context_tokens": 65536,
        "default_layout_version": "minimax-m2.5-v1",
        "default_dtype": "int8",
        "default_block_size": 32,
        "default_lora_id": "base",
        "metadata": {"attention": "mqa", "status": "future-extension"},
        "aliases": ["MiniMaxAI/MiniMax-M2.5-4B", "qwen3.5:4b-instruct"],
    }
    assert round_tripped == definition
    assert from_file == definition
    assert round_tripped.profile.attention_mechanism == AttentionMechanism.MULTI_QUERY

    registry = default_model_profile_registry().with_definition(round_tripped)
    layout = registry.layout_for_model("MiniMaxAI/MiniMax-M2.5-4B", dtype="bf16")

    assert layout.attention_mechanism == AttentionMechanism.MULTI_QUERY
    assert layout.bytes_per_token == 28 * 1 * 128 * 2 * 2


def test_model_profile_definition_rejects_malformed_artifacts():
    valid_record = model_profile_definition_to_record(
        ModelProfileDefinition(profile=QWEN3_4B_INSTRUCT_PROFILE, aliases=("Qwen/Qwen3-4B-Instruct-2507",))
    )

    with pytest.raises(ValueError, match="record_type"):
        model_profile_definition_from_record({**valid_record, "record_type": "document_kv.other.v1"})

    with pytest.raises(ValueError, match=r"model profile record has unsupported keys: \['debug'\]"):
        model_profile_definition_from_record({**valid_record, "debug": {"accepted": False}})

    with pytest.raises(TypeError, match="aliases"):
        ModelProfileDefinition(profile=QWEN3_4B_INSTRUCT_PROFILE, aliases="Qwen/Qwen3-4B")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="metadata"):
        model_profile_definition_from_record({**valid_record, "metadata": {"bad": 1}})

    with pytest.raises(ValueError, match="aliases"):
        model_profile_definition_from_record({**valid_record, "aliases": [""]})

    with pytest.raises(ValueError, match="object"):
        model_profile_definition_from_record(["not", "an", "object"])  # type: ignore[arg-type]


def test_model_profile_registry_rejects_invalid_entries():
    with pytest.raises(TypeError, match="aliases must be an iterable of strings"):
        default_model_profile_registry().with_profile(QWEN3_4B_INSTRUCT_PROFILE, aliases="abc")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="aliases must be an iterable of strings"):
        default_model_profile_registry().with_profile(QWEN3_4B_INSTRUCT_PROFILE, aliases=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="aliases must be strings"):
        default_model_profile_registry().with_profile(QWEN3_4B_INSTRUCT_PROFILE, aliases=(1,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="aliases must be non-empty"):
        default_model_profile_registry().with_profile(QWEN3_4B_INSTRUCT_PROFILE, aliases=("",))

    with pytest.raises(ValueError, match="aliases must be unique"):
        default_model_profile_registry().with_profile(
            QWEN3_4B_INSTRUCT_PROFILE,
            aliases=("Qwen/Qwen3-4B", "Qwen/Qwen3-4B"),
        )

    with pytest.raises(TypeError, match="profile must be a KVModelProfile"):
        default_model_profile_registry().with_profile(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="aliases must be strings"):
        ModelProfileRegistry({1: QWEN3_4B_INSTRUCT_PROFILE})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="KVModelProfile"):
        ModelProfileRegistry({"bad": object()})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="model_id must be a string"):
        default_model_profile_registry().get(1)  # type: ignore[arg-type]


def test_model_profile_rejects_wrong_constructor_field_types():
    kwargs = {
        "model_id": "toy",
        "architecture": "ToyForCausalLM",
        "num_layers": 2,
        "num_query_heads": 4,
        "num_kv_heads": 2,
        "head_size": 8,
        "max_context_tokens": 1024,
        "default_layout_version": "toy-v1",
        "default_dtype": "int8",
        "default_block_size": 16,
        "default_lora_id": "base",
    }

    with pytest.raises(ValueError, match="model_id"):
        KVModelProfile(**{**kwargs, "model_id": 1})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_layers"):
        KVModelProfile(**{**kwargs, "num_layers": True})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="head_size"):
        KVModelProfile(**{**kwargs, "head_size": "128"})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="default_block_size"):
        KVModelProfile(**{**kwargs, "default_block_size": 0})

    with pytest.raises(ValueError, match="default_dtype"):
        KVModelProfile(**{**kwargs, "default_dtype": 1})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="default_lora_id"):
        KVModelProfile(**{**kwargs, "default_lora_id": 1})  # type: ignore[arg-type]


def test_layout_overrides_preserve_invalid_explicit_values_for_validation():
    with pytest.raises(ValueError, match="Unsupported KV dtype"):
        QWEN3_4B_INSTRUCT_PROFILE.to_layout(dtype="")

    with pytest.raises(ValueError, match="model_id must be non-empty"):
        QWEN3_4B_INSTRUCT_PROFILE.to_layout(model_id="")

    with pytest.raises(ValueError, match="lora_id must be non-empty"):
        QWEN3_4B_INSTRUCT_PROFILE.to_layout(lora_id="")

    with pytest.raises(ValueError, match="layout_version must be non-empty"):
        QWEN3_4B_INSTRUCT_PROFILE.to_layout(layout_version="")

    with pytest.raises(ValueError, match="block_size must be positive"):
        QWEN3_4B_INSTRUCT_PROFILE.to_layout(block_size=0)

    with pytest.raises(ValueError, match="kv_stride_bytes must be positive"):
        QWEN3_4B_INSTRUCT_PROFILE.to_layout(kv_stride_bytes=0)

    with pytest.raises(ValueError, match="smaller than"):
        QWEN3_4B_INSTRUCT_PROFILE.to_layout(dtype="bf16", kv_stride_bytes=255)
    with pytest.raises(ValueError, match="smaller than"):
        QWEN3_4B_INSTRUCT_PROFILE.bytes_per_token("bf16", kv_stride_bytes=255)

    with pytest.raises(ValueError, match="multiple"):
        QWEN3_4B_INSTRUCT_PROFILE.to_layout(dtype="bf16", kv_stride_bytes=257)
    with pytest.raises(ValueError, match="multiple"):
        QWEN3_4B_INSTRUCT_PROFILE.bytes_per_token("bf16", kv_stride_bytes=257)


def test_profile_supports_mha_and_mqa_attention_modes():
    mha = KVModelProfile(
        model_id="toy-mha",
        architecture="ToyForCausalLM",
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=4,
        head_size=8,
        max_context_tokens=1024,
        default_layout_version="toy-v1",
    )
    mqa = KVModelProfile(
        model_id="toy-mqa",
        architecture="ToyForCausalLM",
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=8,
        max_context_tokens=1024,
        default_layout_version="toy-v1",
    )

    assert mha.attention_mechanism == AttentionMechanism.MULTI_HEAD
    assert mha.query_heads_per_kv_head == 1
    assert mha.to_layout().attention_mechanism == AttentionMechanism.MULTI_HEAD
    assert mqa.attention_mechanism == AttentionMechanism.MULTI_QUERY
    assert mqa.query_heads_per_kv_head == 4
    assert mqa.to_layout().attention_mechanism == AttentionMechanism.MULTI_QUERY
    assert mqa.to_layout(shares_kv_storage=False).storage_layout == KVStorageLayout.SEPARATE_KEY_VALUE


def test_profile_validation_rejects_invalid_attention_geometry():
    with pytest.raises(ValueError, match="divisible"):
        KVModelProfile(
            model_id="bad-gqa",
            architecture="Toy",
            num_layers=2,
            num_query_heads=5,
            num_kv_heads=2,
            head_size=8,
            max_context_tokens=1024,
            default_layout_version="toy-v1",
        )


def test_profile_validation_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Unsupported KV dtype"):
        KVModelProfile(
            model_id="bad-dtype",
            architecture="Toy",
            num_layers=2,
            num_query_heads=4,
            num_kv_heads=4,
            head_size=8,
            max_context_tokens=1024,
            default_layout_version="toy-v1",
            default_dtype="nf4",
        )
    with pytest.raises(ValueError, match="Unsupported KV dtype"):
        QWEN3_4B_INSTRUCT_PROFILE.bytes_per_token("")
    with pytest.raises(ValueError, match="Unsupported KV dtype"):
        dtype_byte_width("nf4")


def test_unknown_profile_error_lists_supported_aliases():
    with pytest.raises(KeyError, match="supported profiles"):
        get_model_profile("missing-model")
