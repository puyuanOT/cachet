import json

import pytest

from document_kv_cache.engine_probe import EngineKVProbeFactoryContext
from document_kv_cache.engine_probe import load_engine_kv_probe_factory
from document_kv_cache.engine_adapters import ServingBackend
from document_kv_cache.native_probe_factories import (
    NATIVE_PROBE_FACTORIES_RECORD_TYPE,
    NativeProbeFactoryInspection,
    NativeProbeFactoryUnavailable,
    SGLANG_NATIVE_PROBE_FACTORY,
    VLLM_NATIVE_PROBE_FACTORY,
    builtin_native_probe_factories_to_record,
    builtin_native_probe_factory_path,
    inspect_builtin_native_probe_factories,
    inspect_builtin_native_probe_factory,
    main,
    native_probe_factory_inspection_to_record,
    sglang_native_probe_factory,
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
    assert inspection.supported is False
    assert inspection.reason

    record = native_probe_factory_inspection_to_record(inspection)
    assert record["backend"] == "vllm"
    assert record["factory_path"] == VLLM_NATIVE_PROBE_FACTORY
    assert record["supported"] is False
    assert "reason" in record
    assert record["serving_environment_profile"] == serving_environment_profile_to_record(
        VLLM_SERVING_ENVIRONMENT_PROFILE
    )


def test_builtin_native_probe_factories_record_includes_required_backends():
    inspections = inspect_builtin_native_probe_factories()
    assert {inspection.backend.value for inspection in inspections} == {"vllm", "sglang"}

    record = builtin_native_probe_factories_to_record()
    assert record["record_type"] == NATIVE_PROBE_FACTORIES_RECORD_TYPE
    assert {factory["backend"] for factory in record["factories"]} == {"vllm", "sglang"}
    assert all(factory["supported"] is False for factory in record["factories"])
    profile_by_backend = {
        factory["backend"]: factory["serving_environment_profile"]
        for factory in record["factories"]
    }
    assert profile_by_backend == {
        "vllm": serving_environment_profile_to_record(VLLM_SERVING_ENVIRONMENT_PROFILE),
        "sglang": serving_environment_profile_to_record(SGLANG_SERVING_ENVIRONMENT_PROFILE),
    }


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
