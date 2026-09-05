import pytest

from document_kv_cache.artifact_identity import (
    ArtifactIdentity,
    RuntimeCompatibilityHandshake,
    RuntimeIdentity,
    TokenContract,
    UNRESOLVED_IDENTITY,
    method_config_digest,
    token_ids_digest,
)


def _artifact(**overrides):
    values = {
        "method_id": "vanilla_prefill",
        "method_version": "1",
        "method_config_digest": method_config_digest({"pre_rope": False}),
        "model_id": "qwen3:4b-instruct",
        "model_revision": "model-sha",
        "tokenizer_id": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_revision": "tokenizer-sha",
        "lora_id": "none",
        "prompt_template_version": "v1",
        "layout_version": "qwen3-v1",
        "kv_dtype": "bfloat16",
        "block_size": 16,
        "payload_axis_order": "token_major",
    }
    values.update(overrides)
    return ArtifactIdentity(**values)


def _runtime(**overrides):
    artifact = _artifact()
    values = {
        "model_id": artifact.model_id,
        "model_revision": artifact.model_revision,
        "tokenizer_id": artifact.tokenizer_id,
        "tokenizer_revision": artifact.tokenizer_revision,
        "lora_id": artifact.lora_id,
        "prompt_template_version": artifact.prompt_template_version,
        "layout_version": artifact.layout_version,
        "kv_dtype": artifact.kv_dtype,
        "block_size": artifact.block_size,
        "payload_axis_order": artifact.payload_axis_order,
        "key_position_encoding": artifact.key_position_encoding,
        "rope_theta": artifact.rope_theta,
        "rope_rotary_dim": artifact.rope_rotary_dim,
    }
    values.update(overrides)
    return RuntimeIdentity(**values)


def test_artifact_identity_is_stable_and_round_trips():
    artifact = _artifact()

    assert len(artifact.artifact_id) == 64
    assert ArtifactIdentity.from_record(artifact.to_record()) == artifact
    assert _artifact().artifact_id == artifact.artifact_id


def test_method_config_digest_is_key_order_independent():
    assert method_config_digest({"a": 1, "b": False}) == method_config_digest({"b": False, "a": 1})


def test_token_contract_binds_exact_token_sequence():
    contract = TokenContract.from_token_ids(
        [1, 2, 3],
        tokenizer_id="tokenizer",
        tokenizer_revision="revision",
        add_special_tokens=False,
        prompt_template_version="v1",
    )

    assert contract.token_ids_digest == token_ids_digest([1, 2, 3])
    assert TokenContract.from_record(contract.to_record()) == contract
    assert contract.verifies([1, 2, 3])
    assert not contract.verifies([1, 3, 2])
    with pytest.raises(ValueError, match="do not satisfy token contract"):
        contract.require_match([1, 2, 4])


def test_runtime_handshake_accepts_exact_identity():
    handshake = RuntimeCompatibilityHandshake.compare(_artifact(), _runtime())

    assert handshake.compatible
    assert handshake.issues == ()
    handshake.require_compatible()


def test_runtime_handshake_reports_every_mismatch():
    handshake = RuntimeCompatibilityHandshake.compare(
        _artifact(),
        _runtime(model_revision="different", block_size=32),
    )

    assert not handshake.compatible
    assert {issue.field for issue in handshake.issues} == {"model_revision", "block_size"}
    with pytest.raises(ValueError, match="model_revision"):
        handshake.require_compatible()


def test_runtime_handshake_binds_rope_position_identity():
    artifact = _artifact(
        key_position_encoding="pre_rope",
        rope_theta=10_000.0,
        rope_rotary_dim=128,
    )
    runtime = _runtime(
        key_position_encoding="pre_rope",
        rope_theta=500_000.0,
        rope_rotary_dim=128,
    )

    handshake = RuntimeCompatibilityHandshake.compare(
        artifact,
        runtime,
    )

    assert not handshake.compatible
    assert {issue.field for issue in handshake.issues} == {"rope_theta"}


def test_runtime_handshake_rejects_unresolved_identity_by_default():
    artifact = _artifact(model_revision=UNRESOLVED_IDENTITY)
    runtime = _runtime(model_revision=UNRESOLVED_IDENTITY)

    assert not RuntimeCompatibilityHandshake.compare(artifact, runtime).compatible
    assert RuntimeCompatibilityHandshake.compare(
        artifact,
        runtime,
        reject_unresolved=False,
    ).compatible
