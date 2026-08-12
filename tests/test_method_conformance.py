from __future__ import annotations

import json

from document_kv_cache.method_conformance import (
    inspect_method_conformance,
    main,
    method_conformance_to_record,
)
from document_kv_cache.methods import method_spec
from document_kv_cache.models import CacheGenerationMethod


def test_builtin_artifact_method_loads_factory_without_model_instantiation() -> None:
    result = inspect_method_conformance(
        method_spec(CacheGenerationMethod.VANILLA_PREFILL)
    )

    assert result.ok
    assert result.factory_loaded
    assert not result.generator_instantiated
    record = method_conformance_to_record(result)
    assert record["method"]["reuse_plan"]["method_id"] == "vanilla_prefill"


def test_unimplemented_method_fails_closed_unless_explicitly_allowed() -> None:
    method = method_spec(CacheGenerationMethod.CACHEBLEND)

    assert not inspect_method_conformance(method).ok
    assert inspect_method_conformance(method, allow_unimplemented=True).ok


def test_engine_native_method_conforms_without_generator() -> None:
    result = inspect_method_conformance(method_spec(CacheGenerationMethod.LMCACHE))

    assert result.ok
    assert not result.factory_loaded


def test_method_conformance_cli_writes_machine_readable_record(tmp_path) -> None:
    output = tmp_path / "conformance.json"

    assert main(["--method-id", "lmcache", "--output-json", str(output)]) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["method"]["method_id"] == "lmcache"
