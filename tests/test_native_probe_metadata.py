import pytest

from document_kv_cache.engine_adapters import ServingBackend
from document_kv_cache._native_probe_metadata import (
    SGLANG_CONNECTOR_FACTORY_METADATA_EXAMPLE,
    SGLANG_NATIVE_PROBE_DELEGATE_FACTORY,
    VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
    VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    validate_known_native_delegate_metadata,
)


@pytest.mark.parametrize(
    ("backend", "delegate_factory", "metadata"),
    [
        (
            ServingBackend.VLLM,
            VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
            VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
        ),
        (
            ServingBackend.SGLANG,
            SGLANG_NATIVE_PROBE_DELEGATE_FACTORY,
            SGLANG_CONNECTOR_FACTORY_METADATA_EXAMPLE,
        ),
    ],
)
def test_validate_known_native_delegate_metadata_accepts_real_connector_factories(
    backend,
    delegate_factory,
    metadata,
):
    validate_known_native_delegate_metadata(
        backend=backend,
        native_probe_delegate_factory=delegate_factory,
        metadata=(metadata,),
        label="engine probe target",
        backend_field_label="expected_backend",
    )


@pytest.mark.parametrize(
    "metadata",
    [
        "vllm_kv_injection.connector_factory=module:factory",
        "vllm_kv_injection.connector_factory=module:factory ",
        "vllm_kv_injection.connector_factory=company_vllm_patch_probe",
    ],
)
def test_validate_known_native_delegate_metadata_rejects_placeholder_connector_factories(metadata):
    with pytest.raises(ValueError, match="real module:attribute connector factory"):
        validate_known_native_delegate_metadata(
            backend=ServingBackend.VLLM,
            native_probe_delegate_factory=VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
            metadata=(metadata,),
            label="engine probe target",
            backend_field_label="expected_backend",
        )


def test_validate_known_native_delegate_metadata_uses_caller_backend_label():
    with pytest.raises(ValueError, match="but backend is vllm"):
        validate_known_native_delegate_metadata(
            backend=ServingBackend.VLLM,
            native_probe_delegate_factory=SGLANG_NATIVE_PROBE_DELEGATE_FACTORY,
            metadata=(SGLANG_CONNECTOR_FACTORY_METADATA_EXAMPLE,),
            label="release-safe engine_probe_targets",
            backend_field_label="backend",
        )
