import json

import pytest

import document_kv_cache.native_probe_factories as native_probe_factories
from document_kv_cache.engine_probe import EngineKVProbeFactoryContext
from document_kv_cache.engine_probe import load_engine_kv_probe_factory
from document_kv_cache.engine_adapters import (
    ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
    ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
    ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE,
    ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION,
    ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE,
    ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION,
    PayloadMode,
    ServingBackend,
)
from document_kv_cache.native_probe_factories import (
    NATIVE_PROBE_FACTORIES_RECORD_TYPE,
    NATIVE_PROBE_DELEGATE_CONTRACT_ATTR,
    NATIVE_PROBE_DELEGATE_CONTRACT_MODULE_ATTR,
    NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR,
    NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR,
    NativeProbeFactoryInspection,
    NativeProbeFactoryUnavailable,
    SGLANG_NATIVE_PROBE_FACTORY,
    SGLANG_NATIVE_PROBE_DELEGATE_ENV,
    VLLM_NATIVE_PROBE_FACTORY,
    VLLM_NATIVE_PROBE_DELEGATE_ENV,
    builtin_native_probe_factories_to_record,
    builtin_native_probe_factory_path,
    inspect_builtin_native_probe_factories,
    inspect_builtin_native_probe_factory,
    main,
    native_probe_adapter_contract_to_record,
    native_probe_factory_inspection_to_record,
    native_probe_factories_record_issues,
    native_probe_runtime_contract_to_record,
    sglang_native_probe_factory,
    validate_native_probe_factories_record,
    vllm_native_probe_factory,
    write_builtin_native_probe_factories_record_json,
)
from document_kv_cache.serving_env import (
    SGLANG_SERVING_ENVIRONMENT_PROFILE,
    VLLM_SERVING_ENVIRONMENT_PROFILE,
    serving_environment_profile_to_record,
)


class DummyPlan:
    request_id = "req-1"


def context(backend: ServingBackend | str) -> EngineKVProbeFactoryContext:
    return EngineKVProbeFactoryContext(
        backend=backend,
        handoff_record={},
        plan=DummyPlan(),  # type: ignore[arg-type]
        payload_source_uri="/tmp/payload.kv",
    )


def mark_backend_packages_installed(monkeypatch, *, version: str = "0.23.0") -> None:
    def fake_version(package_name: str) -> str:
        if package_name in {"vllm", "sglang"}:
            return version
        raise native_probe_factories.metadata.PackageNotFoundError(package_name)

    def fake_find_spec(package_name: str):
        return object() if package_name in {"vllm", "sglang"} else None

    monkeypatch.setattr(native_probe_factories.metadata, "version", fake_version)
    monkeypatch.setattr(native_probe_factories.util, "find_spec", fake_find_spec)


def write_delegate_factory_module(
    tmp_path,
    monkeypatch,
    *,
    module_name: str,
    backend: ServingBackend | str = ServingBackend.VLLM,
    contract: dict | None = None,
    runtime_contract: dict | None = None,
    include_runtime_contract: bool = True,
) -> str:
    contract_literal = repr(native_probe_adapter_contract_to_record() if contract is None else contract)
    runtime_contract_literal = repr(
        native_probe_runtime_contract_to_record(backend) if runtime_contract is None else runtime_contract
    )
    runtime_contract_declaration = (
        f"{NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR} = {runtime_contract_literal}"
        if include_runtime_contract
        else ""
    )
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        f"""
{NATIVE_PROBE_DELEGATE_CONTRACT_MODULE_ATTR} = {contract_literal}
{runtime_contract_declaration}


class Probe:
    def __init__(self, backend):
        self.backend = backend


def build_probe(context):
    return Probe(context.backend.value)
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_probe"


def write_unhashable_delegate_factory_module(tmp_path, monkeypatch, *, module_name: str) -> str:
    contract_literal = repr(native_probe_adapter_contract_to_record())
    runtime_contract_literal = repr(native_probe_runtime_contract_to_record(ServingBackend.VLLM))
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        f"""
{NATIVE_PROBE_DELEGATE_CONTRACT_MODULE_ATTR} = {contract_literal}
{NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR} = {runtime_contract_literal}


class Probe:
    def __init__(self, backend):
        self.backend = backend


class ProbeFactory:
    def __call__(self, context):
        return Probe(context.backend.value)

    def __eq__(self, other):
        return self is other


build_probe = ProbeFactory()
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_probe"


def write_callable_contract_delegate_factory_module(tmp_path, monkeypatch, *, module_name: str) -> str:
    contract_literal = repr(native_probe_adapter_contract_to_record())
    runtime_contract_literal = repr(native_probe_runtime_contract_to_record(ServingBackend.VLLM))
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        f"""
class Probe:
    def __init__(self, backend):
        self.backend = backend


def build_probe(context):
    return Probe(context.backend.value)


build_probe.{NATIVE_PROBE_DELEGATE_CONTRACT_ATTR} = {contract_literal}
build_probe.{NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR} = {runtime_contract_literal}
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_probe"


def write_contractless_delegate_factory_module(tmp_path, monkeypatch, *, module_name: str) -> str:
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
class Probe:
    pass


def build_probe(context):
    return Probe()
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{module_name}:build_probe"


def test_builtin_native_probe_factory_paths_are_public_document_paths():
    assert builtin_native_probe_factory_path("vllm") == VLLM_NATIVE_PROBE_FACTORY
    assert builtin_native_probe_factory_path(ServingBackend.SGLANG) == SGLANG_NATIVE_PROBE_FACTORY
    assert VLLM_NATIVE_PROBE_FACTORY == "document_kv_cache.native_probe_factories:vllm_native_probe_factory"
    assert SGLANG_NATIVE_PROBE_FACTORY == "document_kv_cache.native_probe_factories:sglang_native_probe_factory"

    with pytest.raises(ValueError, match="Unsupported serving backend"):
        builtin_native_probe_factory_path("triton")


def test_builtin_native_probe_factory_paths_are_loadable():
    assert load_engine_kv_probe_factory(VLLM_NATIVE_PROBE_FACTORY) is vllm_native_probe_factory
    assert load_engine_kv_probe_factory(SGLANG_NATIVE_PROBE_FACTORY) is sglang_native_probe_factory


def test_inspect_builtin_native_probe_factory_reports_fail_closed_status():
    inspection = inspect_builtin_native_probe_factory("vllm")
    assert isinstance(inspection, NativeProbeFactoryInspection)
    assert inspection.backend == ServingBackend.VLLM
    assert inspection.factory_path == VLLM_NATIVE_PROBE_FACTORY
    assert inspection.package_name == "vllm"
    assert inspection.delegate_factory_path is None
    assert inspection.delegate_adapter_contract is None
    assert inspection.delegate_adapter_contract_valid is False
    assert inspection.delegate_runtime_contract is None
    assert inspection.delegate_runtime_contract_valid is False
    assert inspection.supported is False
    assert inspection.reason

    record = native_probe_factory_inspection_to_record(inspection)
    assert record["backend"] == "vllm"
    assert record["factory_path"] == VLLM_NATIVE_PROBE_FACTORY
    assert record["delegate_factory_path"] is None
    assert record["delegate_adapter_contract"] is None
    assert record["delegate_adapter_contract_valid"] is False
    assert record["delegate_runtime_contract"] is None
    assert record["delegate_runtime_contract_valid"] is False
    assert record["supported"] is False
    assert "reason" in record
    assert record["adapter_contract"] == native_probe_adapter_contract_to_record()
    assert record["serving_environment_profile"] == serving_environment_profile_to_record(
        VLLM_SERVING_ENVIRONMENT_PROFILE
    )


def test_native_probe_factory_inspection_preserves_previous_positional_signature():
    inspection = NativeProbeFactoryInspection(
        ServingBackend.VLLM,
        VLLM_NATIVE_PROBE_FACTORY,
        "vllm",
        False,
        None,
        False,
        "vllm is not installed",
    )

    assert inspection.supported is False
    assert inspection.reason == "vllm is not installed"
    assert inspection.delegate_factory_path is None
    assert inspection.delegate_adapter_contract is None
    assert inspection.delegate_adapter_contract_valid is False
    assert inspection.delegate_runtime_contract is None
    assert inspection.delegate_runtime_contract_valid is False


def test_inspect_builtin_native_probe_factory_delegates_when_backend_and_adapter_are_available(
    tmp_path,
    monkeypatch,
):
    mark_backend_packages_installed(monkeypatch)
    vllm_delegate_path = write_delegate_factory_module(
        tmp_path,
        monkeypatch,
        module_name="vllm_delegate_probe",
    )
    sglang_delegate_path = write_delegate_factory_module(
        tmp_path,
        monkeypatch,
        module_name="sglang_delegate_probe",
        backend=ServingBackend.SGLANG,
    )
    monkeypatch.setenv(VLLM_NATIVE_PROBE_DELEGATE_ENV, vllm_delegate_path)
    monkeypatch.setenv(SGLANG_NATIVE_PROBE_DELEGATE_ENV, sglang_delegate_path)

    inspection = inspect_builtin_native_probe_factory("vllm")
    assert inspection.supported is True
    assert inspection.delegate_factory_path == vllm_delegate_path
    assert inspection.delegate_adapter_contract == native_probe_adapter_contract_to_record()
    assert inspection.delegate_adapter_contract_valid is True
    assert inspection.delegate_runtime_contract == native_probe_runtime_contract_to_record("vllm")
    assert inspection.delegate_runtime_contract_valid is True
    assert "declares the Document KV adapter and runtime contracts" in inspection.reason

    probe = vllm_native_probe_factory(context(ServingBackend.VLLM))
    assert probe.backend == "vllm"

    record = builtin_native_probe_factories_to_record()
    assert {factory["delegate_factory_path"] for factory in record["factories"]} == {
        vllm_delegate_path,
        sglang_delegate_path,
    }
    assert all(factory["supported"] is True for factory in record["factories"])
    assert all(
        factory["delegate_adapter_contract"] == native_probe_adapter_contract_to_record()
        for factory in record["factories"]
    )
    assert all(factory["delegate_adapter_contract_valid"] is True for factory in record["factories"])
    assert {
        factory["backend"]: factory["delegate_runtime_contract"]
        for factory in record["factories"]
    } == {
        "vllm": native_probe_runtime_contract_to_record("vllm"),
        "sglang": native_probe_runtime_contract_to_record("sglang"),
    }
    assert all(factory["delegate_runtime_contract_valid"] is True for factory in record["factories"])
    validate_native_probe_factories_record(record)


def test_inspect_builtin_native_probe_factory_accepts_callable_declared_delegate_contract(
    tmp_path,
    monkeypatch,
):
    mark_backend_packages_installed(monkeypatch)
    delegate_path = write_callable_contract_delegate_factory_module(
        tmp_path,
        monkeypatch,
        module_name="vllm_callable_contract_delegate_probe",
    )
    monkeypatch.setenv(VLLM_NATIVE_PROBE_DELEGATE_ENV, delegate_path)

    inspection = inspect_builtin_native_probe_factory("vllm")

    assert inspection.supported is True
    assert inspection.delegate_adapter_contract_valid is True
    assert inspection.delegate_adapter_contract == native_probe_adapter_contract_to_record()
    assert inspection.delegate_runtime_contract_valid is True
    assert inspection.delegate_runtime_contract == native_probe_runtime_contract_to_record("vllm")


def test_inspect_builtin_native_probe_factory_requires_declared_delegate_contract(
    tmp_path,
    monkeypatch,
):
    mark_backend_packages_installed(monkeypatch)
    delegate_path = write_contractless_delegate_factory_module(
        tmp_path,
        monkeypatch,
        module_name="vllm_contractless_delegate_probe",
    )
    monkeypatch.setenv(VLLM_NATIVE_PROBE_DELEGATE_ENV, delegate_path)

    inspection = inspect_builtin_native_probe_factory("vllm")

    assert inspection.supported is False
    assert inspection.delegate_factory_path == delegate_path
    assert inspection.delegate_adapter_contract is None
    assert inspection.delegate_adapter_contract_valid is False
    assert "must declare" in inspection.reason
    with pytest.raises(NativeProbeFactoryUnavailable, match="must declare"):
        vllm_native_probe_factory(context(ServingBackend.VLLM))


def test_inspect_builtin_native_probe_factory_requires_declared_runtime_contract(
    tmp_path,
    monkeypatch,
):
    mark_backend_packages_installed(monkeypatch)
    delegate_path = write_delegate_factory_module(
        tmp_path,
        monkeypatch,
        module_name="vllm_adapter_only_delegate_probe",
        include_runtime_contract=False,
    )
    monkeypatch.setenv(VLLM_NATIVE_PROBE_DELEGATE_ENV, delegate_path)

    inspection = inspect_builtin_native_probe_factory("vllm")

    assert inspection.supported is False
    assert inspection.delegate_factory_path == delegate_path
    assert inspection.delegate_adapter_contract == native_probe_adapter_contract_to_record()
    assert inspection.delegate_adapter_contract_valid is True
    assert inspection.delegate_runtime_contract is None
    assert inspection.delegate_runtime_contract_valid is False
    assert NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR in inspection.reason
    with pytest.raises(NativeProbeFactoryUnavailable, match=NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR):
        vllm_native_probe_factory(context(ServingBackend.VLLM))


def test_inspect_builtin_native_probe_factory_rejects_mismatched_runtime_contract(
    tmp_path,
    monkeypatch,
):
    mark_backend_packages_installed(monkeypatch)
    wrong_runtime_contract = {
        **native_probe_runtime_contract_to_record("vllm"),
        "runtime": "debug-only-runtime",
    }
    delegate_path = write_delegate_factory_module(
        tmp_path,
        monkeypatch,
        module_name="vllm_wrong_runtime_contract_delegate_probe",
        runtime_contract=wrong_runtime_contract,
    )
    monkeypatch.setenv(VLLM_NATIVE_PROBE_DELEGATE_ENV, delegate_path)

    inspection = inspect_builtin_native_probe_factory("vllm")

    assert inspection.supported is False
    assert inspection.delegate_adapter_contract_valid is True
    assert inspection.delegate_runtime_contract == wrong_runtime_contract
    assert inspection.delegate_runtime_contract_valid is False
    assert "incompatible native runtime contract" in inspection.reason
    assert "runtime must match" in inspection.reason
    with pytest.raises(NativeProbeFactoryUnavailable, match="runtime must match"):
        vllm_native_probe_factory(context(ServingBackend.VLLM))


def test_inspect_builtin_native_probe_factory_rejects_mismatched_delegate_contract(
    tmp_path,
    monkeypatch,
):
    mark_backend_packages_installed(monkeypatch)
    wrong_contract = {
        **native_probe_adapter_contract_to_record(),
        "payload_mode": PayloadMode.SEGMENTED.value,
    }
    delegate_path = write_delegate_factory_module(
        tmp_path,
        monkeypatch,
        module_name="vllm_wrong_contract_delegate_probe",
        contract=wrong_contract,
    )
    monkeypatch.setenv(VLLM_NATIVE_PROBE_DELEGATE_ENV, delegate_path)

    inspection = inspect_builtin_native_probe_factory("vllm")

    assert inspection.supported is False
    assert inspection.delegate_adapter_contract == wrong_contract
    assert inspection.delegate_adapter_contract_valid is False
    assert "incompatible Document KV adapter contract" in inspection.reason
    assert "payload_mode must match" in inspection.reason
    with pytest.raises(NativeProbeFactoryUnavailable, match="payload_mode must match"):
        vllm_native_probe_factory(context(ServingBackend.VLLM))


def test_inspect_builtin_native_probe_factory_accepts_unhashable_callable_delegate(
    tmp_path,
    monkeypatch,
):
    mark_backend_packages_installed(monkeypatch)
    delegate_path = write_unhashable_delegate_factory_module(
        tmp_path,
        monkeypatch,
        module_name="vllm_unhashable_delegate_probe",
    )
    monkeypatch.setenv(VLLM_NATIVE_PROBE_DELEGATE_ENV, delegate_path)

    inspection = inspect_builtin_native_probe_factory("vllm")

    assert inspection.supported is True
    assert inspection.delegate_factory_path == delegate_path
    assert inspection.delegate_adapter_contract_valid is True
    probe = vllm_native_probe_factory(context(ServingBackend.VLLM))
    assert probe.backend == "vllm"


def test_inspect_builtin_native_probe_factory_reports_unloadable_delegate(monkeypatch):
    mark_backend_packages_installed(monkeypatch)
    monkeypatch.setenv(VLLM_NATIVE_PROBE_DELEGATE_ENV, "missing_delegate_module:build_probe")

    inspection = inspect_builtin_native_probe_factory("vllm")

    assert inspection.supported is False
    assert inspection.delegate_factory_path == "missing_delegate_module:build_probe"
    assert "delegate native probe factory" in inspection.reason
    assert "unavailable" in inspection.reason
    with pytest.raises(NativeProbeFactoryUnavailable, match="delegate native probe factory"):
        vllm_native_probe_factory(context(ServingBackend.VLLM))


def test_inspect_builtin_native_probe_factory_requires_importable_backend_for_delegate_support(
    tmp_path,
    monkeypatch,
):
    def fake_version(package_name: str) -> str:
        if package_name == "vllm":
            return "0.23.0"
        raise native_probe_factories.metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr(native_probe_factories.metadata, "version", fake_version)
    monkeypatch.setattr(native_probe_factories.util, "find_spec", lambda package_name: None)
    delegate_path = write_delegate_factory_module(
        tmp_path,
        monkeypatch,
        module_name="vllm_metadata_only_delegate_probe",
    )
    monkeypatch.setenv(VLLM_NATIVE_PROBE_DELEGATE_ENV, delegate_path)

    inspection = inspect_builtin_native_probe_factory("vllm")

    assert inspection.supported is False
    assert inspection.package_importable is False
    assert inspection.package_version == "0.23.0"
    assert inspection.delegate_factory_path == delegate_path
    assert "package metadata is available but the package is not importable" in inspection.reason
    with pytest.raises(NativeProbeFactoryUnavailable, match="not importable"):
        vllm_native_probe_factory(context(ServingBackend.VLLM))
    assert native_probe_factories_record_issues(builtin_native_probe_factories_to_record()) == ()


@pytest.mark.parametrize(
    ("backend", "env_name", "delegate_path", "factory"),
    [
        (ServingBackend.VLLM, VLLM_NATIVE_PROBE_DELEGATE_ENV, VLLM_NATIVE_PROBE_FACTORY, vllm_native_probe_factory),
        (
            ServingBackend.VLLM,
            VLLM_NATIVE_PROBE_DELEGATE_ENV,
            "document_kv_cache.native_probe_factories.vllm_native_probe_factory",
            vllm_native_probe_factory,
        ),
        (
            ServingBackend.VLLM,
            VLLM_NATIVE_PROBE_DELEGATE_ENV,
            "document_kv_cache:vllm_native_probe_factory",
            vllm_native_probe_factory,
        ),
        (
            ServingBackend.VLLM,
            VLLM_NATIVE_PROBE_DELEGATE_ENV,
            "cachet:vllm_native_probe_factory",
            vllm_native_probe_factory,
        ),
        (
            ServingBackend.VLLM,
            VLLM_NATIVE_PROBE_DELEGATE_ENV,
            "restaurant_kv_serving.native_probe_factories:vllm_native_probe_factory",
            vllm_native_probe_factory,
        ),
        (
            ServingBackend.VLLM,
            VLLM_NATIVE_PROBE_DELEGATE_ENV,
            SGLANG_NATIVE_PROBE_FACTORY,
            vllm_native_probe_factory,
        ),
        (
            ServingBackend.VLLM,
            VLLM_NATIVE_PROBE_DELEGATE_ENV,
            "document_kv_cache.native_probe_factories.sglang_native_probe_factory",
            vllm_native_probe_factory,
        ),
        (
            ServingBackend.VLLM,
            VLLM_NATIVE_PROBE_DELEGATE_ENV,
            "restaurant_kv_serving:sglang_native_probe_factory",
            vllm_native_probe_factory,
        ),
        (
            ServingBackend.SGLANG,
            SGLANG_NATIVE_PROBE_DELEGATE_ENV,
            SGLANG_NATIVE_PROBE_FACTORY,
            sglang_native_probe_factory,
        ),
        (
            ServingBackend.SGLANG,
            SGLANG_NATIVE_PROBE_DELEGATE_ENV,
            "document_kv_cache.native_probe_factories.sglang_native_probe_factory",
            sglang_native_probe_factory,
        ),
        (
            ServingBackend.SGLANG,
            SGLANG_NATIVE_PROBE_DELEGATE_ENV,
            "document_kv_cache:sglang_native_probe_factory",
            sglang_native_probe_factory,
        ),
        (
            ServingBackend.SGLANG,
            SGLANG_NATIVE_PROBE_DELEGATE_ENV,
            "cachet:sglang_native_probe_factory",
            sglang_native_probe_factory,
        ),
        (
            ServingBackend.SGLANG,
            SGLANG_NATIVE_PROBE_DELEGATE_ENV,
            "restaurant_kv_serving.native_probe_factories:sglang_native_probe_factory",
            sglang_native_probe_factory,
        ),
        (
            ServingBackend.SGLANG,
            SGLANG_NATIVE_PROBE_DELEGATE_ENV,
            VLLM_NATIVE_PROBE_FACTORY,
            sglang_native_probe_factory,
        ),
        (
            ServingBackend.SGLANG,
            SGLANG_NATIVE_PROBE_DELEGATE_ENV,
            "document_kv_cache.native_probe_factories.vllm_native_probe_factory",
            sglang_native_probe_factory,
        ),
        (
            ServingBackend.SGLANG,
            SGLANG_NATIVE_PROBE_DELEGATE_ENV,
            "restaurant_kv_serving:vllm_native_probe_factory",
            sglang_native_probe_factory,
        ),
    ],
)
def test_inspect_builtin_native_probe_factory_rejects_builtin_delegate_paths(
    monkeypatch,
    backend,
    env_name,
    delegate_path,
    factory,
):
    mark_backend_packages_installed(monkeypatch)
    monkeypatch.setenv(env_name, delegate_path)

    inspection = inspect_builtin_native_probe_factory(backend)

    assert inspection.supported is False
    assert inspection.delegate_factory_path == delegate_path
    assert "points at a built-in Document KV factory" in inspection.reason
    with pytest.raises(NativeProbeFactoryUnavailable, match="built-in Document KV factory"):
        factory(context(backend))


def test_native_probe_adapter_contract_records_required_engine_handoff_versions():
    assert native_probe_adapter_contract_to_record() == {
        "handoff_record_type": ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
        "handoff_schema_version": ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
        "probe_record_type": ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE,
        "probe_schema_version": ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION,
        "actions_record_type": ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE,
        "actions_schema_version": ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION,
        "layout_version": "qwen3-v1",
        "payload_mode": PayloadMode.MERGED.value,
        "requires_native_probe": True,
    }


def test_native_probe_runtime_contract_records_required_backend_lifecycles():
    vllm_contract = native_probe_runtime_contract_to_record("vllm")
    assert vllm_contract["record_type"] == "vllm_kv_injection.kv_connector_v1_contract.v1"
    assert vllm_contract["runtime"] == "vllm-kv-connector-v1"
    assert vllm_contract["handoff_contract"] == native_probe_adapter_contract_to_record()
    assert "get_num_new_matched_tokens" in vllm_contract["required_methods"]
    assert "request_finished" in vllm_contract["required_methods"]
    assert "shutdown" in vllm_contract["optional_methods"]

    sglang_contract = native_probe_runtime_contract_to_record(ServingBackend.SGLANG)
    assert sglang_contract == {
        "record_type": "sglang_kv_injection.runtime_cache_contract.v1",
        "schema_version": 1,
        "runtime": "sglang-runtime-cache",
        "doc_url": "https://docs.sglang.io/docs/advanced_features/hicache_design",
        "required_methods": ["stage", "attach", "release"],
        "optional_methods": [],
        "handoff_contract": native_probe_adapter_contract_to_record(),
    }


def test_builtin_native_probe_factories_record_includes_required_backends():
    inspections = inspect_builtin_native_probe_factories()
    assert {inspection.backend.value for inspection in inspections} == {"vllm", "sglang"}

    record = builtin_native_probe_factories_to_record()
    assert record["record_type"] == NATIVE_PROBE_FACTORIES_RECORD_TYPE
    assert {factory["backend"] for factory in record["factories"]} == {"vllm", "sglang"}
    assert all(factory["supported"] is False for factory in record["factories"])
    assert all(factory["delegate_factory_path"] is None for factory in record["factories"])
    assert all(factory["delegate_adapter_contract"] is None for factory in record["factories"])
    assert all(factory["delegate_adapter_contract_valid"] is False for factory in record["factories"])
    assert all(factory["delegate_runtime_contract"] is None for factory in record["factories"])
    assert all(factory["delegate_runtime_contract_valid"] is False for factory in record["factories"])
    profile_by_backend = {
        factory["backend"]: factory["serving_environment_profile"]
        for factory in record["factories"]
    }
    assert profile_by_backend == {
        "vllm": serving_environment_profile_to_record(VLLM_SERVING_ENVIRONMENT_PROFILE),
        "sglang": serving_environment_profile_to_record(SGLANG_SERVING_ENVIRONMENT_PROFILE),
    }


def test_validate_native_probe_factories_record_accepts_builtin_record():
    record = builtin_native_probe_factories_to_record()

    assert native_probe_factories_record_issues(record) == ()
    validate_native_probe_factories_record(record)


def test_validate_native_probe_factories_record_accepts_legacy_v1_without_delegate_contract_fields():
    record = builtin_native_probe_factories_to_record()
    for factory in record["factories"]:
        del factory["delegate_adapter_contract"]
        del factory["delegate_adapter_contract_valid"]
        del factory["delegate_runtime_contract"]
        del factory["delegate_runtime_contract_valid"]

    assert native_probe_factories_record_issues(record) == ()
    validate_native_probe_factories_record(record)


def test_validate_native_probe_factories_record_reports_malformed_sidecars():
    missing_backend_record = builtin_native_probe_factories_to_record()
    missing_backend_record["factories"] = missing_backend_record["factories"][:1]

    issues = native_probe_factories_record_issues(missing_backend_record)

    assert "native probe factories sidecar backends must match required backends" in issues
    with pytest.raises(ValueError, match="backends must match required backends"):
        validate_native_probe_factories_record(missing_backend_record)

    wrong_path_record = builtin_native_probe_factories_to_record()
    wrong_path_record["factories"][0]["factory_path"] = "downstream:factory"
    wrong_path_issues = native_probe_factories_record_issues(wrong_path_record)

    assert any("factory_path must match the built-in vllm factory path" in issue for issue in wrong_path_issues)
    with pytest.raises(ValueError, match="factory_path must match the built-in vllm factory path"):
        validate_native_probe_factories_record(wrong_path_record)

    wrong_contract_record = builtin_native_probe_factories_to_record()
    wrong_contract_record["factories"][0]["adapter_contract"] = {
        **wrong_contract_record["factories"][0]["adapter_contract"],
        "payload_mode": "segmented",
    }
    wrong_contract_issues = native_probe_factories_record_issues(wrong_contract_record)

    assert any("adapter_contract.payload_mode must match" in issue for issue in wrong_contract_issues)
    with pytest.raises(ValueError, match=r"adapter_contract\.payload_mode must match"):
        validate_native_probe_factories_record(wrong_contract_record)

    wrong_contract_type_record = builtin_native_probe_factories_to_record()
    wrong_contract_type_record["factories"][0]["adapter_contract"] = {
        **wrong_contract_type_record["factories"][0]["adapter_contract"],
        "actions_schema_version": True,
        "requires_native_probe": 1,
    }
    wrong_contract_type_issues = native_probe_factories_record_issues(wrong_contract_type_record)

    assert any(
        "adapter_contract.actions_schema_version must be an integer" in issue
        for issue in wrong_contract_type_issues
    )
    assert any(
        "adapter_contract.requires_native_probe must be boolean" in issue
        for issue in wrong_contract_type_issues
    )
    with pytest.raises(ValueError, match=r"adapter_contract\.actions_schema_version must be an integer"):
        validate_native_probe_factories_record(wrong_contract_type_record)

    inconsistent_supported_record = builtin_native_probe_factories_to_record()
    inconsistent_supported_record["factories"][0] = {
        **inconsistent_supported_record["factories"][0],
        "package_importable": True,
        "package_version": "0.23.0",
        "delegate_adapter_contract": native_probe_adapter_contract_to_record(),
        "delegate_adapter_contract_valid": True,
        "supported": True,
    }
    inconsistent_supported_issues = native_probe_factories_record_issues(inconsistent_supported_record)

    assert any(
        "delegate_factory_path must be non-empty when supported is true" in issue
        for issue in inconsistent_supported_issues
    )
    with pytest.raises(ValueError, match="delegate_factory_path must be non-empty when supported is true"):
        validate_native_probe_factories_record(inconsistent_supported_record)

    reserved_delegate_record = builtin_native_probe_factories_to_record()
    reserved_delegate_record["factories"][0] = {
        **reserved_delegate_record["factories"][0],
        "delegate_factory_path": VLLM_NATIVE_PROBE_FACTORY,
        "package_importable": True,
        "package_version": "0.23.0",
        "delegate_adapter_contract": native_probe_adapter_contract_to_record(),
        "delegate_adapter_contract_valid": True,
        "supported": True,
    }
    reserved_delegate_issues = native_probe_factories_record_issues(reserved_delegate_record)

    assert any(
        "delegate_factory_path must not point at a built-in native probe factory" in issue
        for issue in reserved_delegate_issues
    )
    with pytest.raises(ValueError, match="must not point at a built-in native probe factory"):
        validate_native_probe_factories_record(reserved_delegate_record)

    dotted_reserved_delegate_record = builtin_native_probe_factories_to_record()
    dotted_reserved_delegate_record["factories"][0] = {
        **dotted_reserved_delegate_record["factories"][0],
        "delegate_factory_path": "document_kv_cache.native_probe_factories.vllm_native_probe_factory",
        "package_importable": True,
        "package_version": "0.23.0",
        "delegate_adapter_contract": native_probe_adapter_contract_to_record(),
        "delegate_adapter_contract_valid": True,
        "supported": True,
    }
    dotted_reserved_delegate_issues = native_probe_factories_record_issues(
        dotted_reserved_delegate_record
    )

    assert any(
        "delegate_factory_path must not point at a built-in native probe factory" in issue
        for issue in dotted_reserved_delegate_issues
    )
    with pytest.raises(ValueError, match="must not point at a built-in native probe factory"):
        validate_native_probe_factories_record(dotted_reserved_delegate_record)

    alias_reserved_delegate_record = builtin_native_probe_factories_to_record()
    alias_reserved_delegate_record["factories"][0] = {
        **alias_reserved_delegate_record["factories"][0],
        "delegate_factory_path": "cachet:vllm_native_probe_factory",
        "package_importable": True,
        "package_version": "0.23.0",
        "delegate_adapter_contract": native_probe_adapter_contract_to_record(),
        "delegate_adapter_contract_valid": True,
        "supported": True,
    }
    alias_reserved_delegate_issues = native_probe_factories_record_issues(alias_reserved_delegate_record)

    assert any(
        "delegate_factory_path must not point at a built-in native probe factory" in issue
        for issue in alias_reserved_delegate_issues
    )
    with pytest.raises(ValueError, match="must not point at a built-in native probe factory"):
        validate_native_probe_factories_record(alias_reserved_delegate_record)

    invalid_supported_contract_record = builtin_native_probe_factories_to_record()
    invalid_supported_contract_record["factories"][0] = {
        **invalid_supported_contract_record["factories"][0],
        "delegate_factory_path": "downstream:factory",
        "package_importable": True,
        "package_version": "0.23.0",
        "delegate_adapter_contract": native_probe_adapter_contract_to_record(),
        "delegate_adapter_contract_valid": False,
        "supported": True,
    }
    invalid_supported_contract_issues = native_probe_factories_record_issues(
        invalid_supported_contract_record
    )

    assert any(
        "delegate_adapter_contract_valid must be true when supported is true" in issue
        for issue in invalid_supported_contract_issues
    )
    with pytest.raises(ValueError, match="delegate_adapter_contract_valid must be true"):
        validate_native_probe_factories_record(invalid_supported_contract_record)

    missing_supported_contract_record = builtin_native_probe_factories_to_record()
    missing_supported_contract_record["factories"][0] = {
        **missing_supported_contract_record["factories"][0],
        "delegate_factory_path": "downstream:factory",
        "package_importable": True,
        "package_version": "0.23.0",
        "delegate_adapter_contract_valid": True,
        "supported": True,
    }
    missing_supported_contract_issues = native_probe_factories_record_issues(
        missing_supported_contract_record
    )

    assert any(
        "delegate_adapter_contract must be an object when supported is true" in issue
        for issue in missing_supported_contract_issues
    )
    with pytest.raises(ValueError, match="delegate_adapter_contract must be an object"):
        validate_native_probe_factories_record(missing_supported_contract_record)

    invalid_supported_runtime_contract_record = builtin_native_probe_factories_to_record()
    invalid_supported_runtime_contract_record["factories"][0] = {
        **invalid_supported_runtime_contract_record["factories"][0],
        "delegate_factory_path": "downstream:factory",
        "package_importable": True,
        "package_version": "0.23.0",
        "delegate_adapter_contract": native_probe_adapter_contract_to_record(),
        "delegate_adapter_contract_valid": True,
        "delegate_runtime_contract": native_probe_runtime_contract_to_record("vllm"),
        "delegate_runtime_contract_valid": False,
        "supported": True,
    }
    invalid_supported_runtime_contract_issues = native_probe_factories_record_issues(
        invalid_supported_runtime_contract_record
    )

    assert any(
        "delegate_runtime_contract_valid must be true when supported is true" in issue
        for issue in invalid_supported_runtime_contract_issues
    )
    with pytest.raises(ValueError, match="delegate_runtime_contract_valid must be true"):
        validate_native_probe_factories_record(invalid_supported_runtime_contract_record)

    mismatched_runtime_contract_record = builtin_native_probe_factories_to_record()
    mismatched_runtime_contract_record["factories"][0] = {
        **mismatched_runtime_contract_record["factories"][0],
        "delegate_runtime_contract": {
            **native_probe_runtime_contract_to_record("vllm"),
            "runtime": "debug-only-runtime",
        },
        "delegate_runtime_contract_valid": True,
    }
    mismatched_runtime_contract_issues = native_probe_factories_record_issues(
        mismatched_runtime_contract_record
    )

    assert any(
        "delegate_runtime_contract.runtime must match" in issue
        for issue in mismatched_runtime_contract_issues
    )
    with pytest.raises(ValueError, match=r"delegate_runtime_contract\.runtime must match"):
        validate_native_probe_factories_record(mismatched_runtime_contract_record)


def test_builtin_native_probe_factories_record_writer_and_cli(tmp_path, capsys):
    output_path = tmp_path / "native-probe-factories.json"

    write_builtin_native_probe_factories_record_json(output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == builtin_native_probe_factories_to_record()

    assert main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == written

    cli_output_path = tmp_path / "cli" / "native-probe-factories.json"
    assert main(["--output-json", str(cli_output_path)]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(cli_output_path.read_text(encoding="utf-8")) == written


def test_reserved_factories_reject_backend_mismatch_before_environment_probe():
    with pytest.raises(ValueError, match="vllm native probe factory received 'sglang'"):
        vllm_native_probe_factory(context(ServingBackend.SGLANG))

    with pytest.raises(ValueError, match="sglang native probe factory received 'vllm'"):
        sglang_native_probe_factory(context(ServingBackend.VLLM))

    with pytest.raises(ValueError, match="vllm native probe factory received 'sglang'"):
        vllm_native_probe_factory(context("sglang"))


def test_reserved_factories_raise_unavailable_instead_of_debug_native_probe():
    with pytest.raises(NativeProbeFactoryUnavailable):
        vllm_native_probe_factory(context(ServingBackend.VLLM))

    with pytest.raises(NativeProbeFactoryUnavailable):
        sglang_native_probe_factory(context(ServingBackend.SGLANG))
