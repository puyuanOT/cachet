import pytest

from document_kv_cache.engine_protocol import (
    KVLayout,
    KVPayloadAxisOrder,
    kv_payload_axis_order_from_value,
)


def layout(**overrides) -> KVLayout:
    values = {
        "model_id": "toy-model",
        "lora_id": "base",
        "layout_version": "toy-v1",
        "dtype": "int8",
        "num_layers": 1,
        "block_size": 8,
        "bytes_per_token": 1,
    }
    values.update(overrides)
    return KVLayout(**values)


def test_kv_layout_defaults_to_token_major_payload_axis_order():
    assert layout().payload_axis_order is KVPayloadAxisOrder.TOKEN_MAJOR


def test_kv_layout_normalizes_layer_major_payload_axis_order_to_enum():
    assert layout(payload_axis_order="layer_major").payload_axis_order is KVPayloadAxisOrder.LAYER_MAJOR
    # Whitespace and case variants normalize to the same enum member.
    assert layout(payload_axis_order="  LAYER_MAJOR  ").payload_axis_order is KVPayloadAxisOrder.LAYER_MAJOR
    # Enum members pass through unchanged.
    assert (
        layout(payload_axis_order=KVPayloadAxisOrder.LAYER_MAJOR).payload_axis_order
        is KVPayloadAxisOrder.LAYER_MAJOR
    )


def test_kv_layout_rejects_unknown_payload_axis_order():
    with pytest.raises(ValueError, match="Unsupported payload_axis_order"):
        layout(payload_axis_order="head_major")


def test_kv_payload_axis_order_from_value_normalizes_and_rejects_junk():
    assert kv_payload_axis_order_from_value("token_major") is KVPayloadAxisOrder.TOKEN_MAJOR
    assert kv_payload_axis_order_from_value(" Layer_Major ") is KVPayloadAxisOrder.LAYER_MAJOR
    assert kv_payload_axis_order_from_value(KVPayloadAxisOrder.LAYER_MAJOR) is KVPayloadAxisOrder.LAYER_MAJOR

    with pytest.raises(ValueError, match="Unsupported payload_axis_order 'nonsense'"):
        kv_payload_axis_order_from_value("nonsense")
    with pytest.raises(ValueError, match="custom_field"):
        kv_payload_axis_order_from_value("bogus", field_name="custom_field")
