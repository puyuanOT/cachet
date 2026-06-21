import importlib.util
from importlib import metadata as importlib_metadata
import subprocess
import sys
from pathlib import Path

import pytest

from document_kv_cache.adapter_scaffold import (
    NativeProbeDelegateScaffoldConfig,
    main,
    render_native_probe_delegate_module,
    write_native_probe_delegate_module,
)
from document_kv_cache.engine_adapters import EngineKVReservationAction, ServingBackend
from document_kv_cache.engine_protocol import KVLayout, KVStorageLayout
from document_kv_cache.engine_probe import EngineKVProbeFactoryContext
from document_kv_cache.native_probe_factories import (
    native_probe_adapter_contract_to_record,
    native_probe_runtime_contract_to_record,
)


def _layout() -> KVLayout:
    return KVLayout(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        layout_version="qwen3-v1",
        dtype="int8",
        num_layers=36,
        block_size=16,
        bytes_per_token=73728,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        kv_stride_bytes=128,
        shares_kv_storage=True,
        storage_layout=KVStorageLayout.SHARED_KEY_VALUE,
    )


class DummyPlan:
    request_id = "req-1"


def _context(backend: ServingBackend | str) -> EngineKVProbeFactoryContext:
    return EngineKVProbeFactoryContext(
        backend=backend,
        handoff_record={},
        plan=DummyPlan(),  # type: ignore[arg-type]
        payload_source_uri="/tmp/cachet-payload.kv",
    )


def _import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_native_probe_delegate_module_declares_fail_closed_vllm_contract():
    text = render_native_probe_delegate_module(
        NativeProbeDelegateScaffoldConfig(
            backend="vllm",
            module_name="cachet_vllm_native_probe",
            class_name="ProjectVLLMProbe",
        )
    )

    assert "DOCUMENT_KV_NATIVE_PROBE_CONTRACT = native_probe_adapter_contract_to_record()" in text
    assert 'native_probe_runtime_contract_to_record("vllm")' in text
    assert "class ProjectVLLMProbe" in text
    assert "raise NotImplementedError" in text
    assert "debug_in_memory" not in text


def test_write_native_probe_delegate_module_can_be_imported_and_rejects_wrong_backend(tmp_path):
    output_path = write_native_probe_delegate_module(
        NativeProbeDelegateScaffoldConfig(backend=ServingBackend.VLLM),
        tmp_path / "cachet_vllm_native_probe.py",
    )
    module = _import_module(output_path, "cachet_vllm_native_probe")

    assert module.DOCUMENT_KV_NATIVE_PROBE_CONTRACT == native_probe_adapter_contract_to_record()
    assert module.DOCUMENT_KV_NATIVE_PROBE_RUNTIME_CONTRACT == native_probe_runtime_contract_to_record("vllm")
    assert module.build_probe.document_kv_native_probe_contract == native_probe_adapter_contract_to_record()
    assert module.build_probe.document_kv_native_probe_runtime_contract == native_probe_runtime_contract_to_record(
        "vllm"
    )

    module._detect_engine_version = lambda: "0.23.0"
    result = module.build_probe(_context("vllm"))
    assert result.native_probe is True
    assert result.metadata["document_kv.adapter_scaffold"] == "vllm"

    with pytest.raises(ValueError, match="expected vllm"):
        module.build_probe(_context("sglang"))

    with pytest.raises(NotImplementedError, match="native KV block allocator"):
        result.probe.reserve_kv_blocks(
            EngineKVReservationAction(
                backend=ServingBackend.VLLM,
                request_id="req-1",
                total_blocks=1,
                total_tokens=1,
                estimated_gpu_bytes=73728,
                layout=_layout(),
                adapter_ids=(),
            )
        )


def test_generated_delegate_fails_when_backend_version_is_unavailable(tmp_path, monkeypatch):
    output_path = write_native_probe_delegate_module(
        NativeProbeDelegateScaffoldConfig(backend=ServingBackend.VLLM),
        tmp_path / "cachet_vllm_native_probe.py",
    )
    module = _import_module(output_path, "cachet_vllm_native_probe_missing_version")

    def raise_missing(package_name: str):
        raise importlib_metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr(importlib_metadata, "version", raise_missing)

    with pytest.raises(RuntimeError, match="install vllm"):
        module.build_probe(_context("vllm"))


def test_write_native_probe_delegate_module_requires_overwrite(tmp_path):
    output_path = tmp_path / "cachet_sglang_native_probe.py"
    write_native_probe_delegate_module(NativeProbeDelegateScaffoldConfig(backend="sglang"), output_path)

    with pytest.raises(FileExistsError, match="overwrite"):
        write_native_probe_delegate_module(NativeProbeDelegateScaffoldConfig(backend="sglang"), output_path)

    written = write_native_probe_delegate_module(
        NativeProbeDelegateScaffoldConfig(backend="sglang"),
        output_path,
        overwrite=True,
    )
    assert written == output_path


def test_native_probe_delegate_scaffold_config_validates_identifiers():
    with pytest.raises(ValueError, match="module_name"):
        NativeProbeDelegateScaffoldConfig(backend="vllm", module_name="../probe")

    with pytest.raises(ValueError, match="module_name"):
        NativeProbeDelegateScaffoldConfig(backend="vllm", module_name="")

    with pytest.raises(ValueError, match="class_name"):
        NativeProbeDelegateScaffoldConfig(backend="vllm", class_name="not-a-class")

    with pytest.raises(ValueError, match="class_name"):
        NativeProbeDelegateScaffoldConfig(backend="vllm", class_name="class")

    with pytest.raises(ValueError, match="class_name"):
        NativeProbeDelegateScaffoldConfig(backend="vllm", class_name="")


def test_adapter_scaffold_cli_writes_module(tmp_path):
    output_path = tmp_path / "custom_probe.py"

    assert main(["--backend", "vllm", "--output-file", str(output_path), "--module-name", "custom_probe"]) == 0
    assert "document_kv.adapter_module" in output_path.read_text(encoding="utf-8")


def test_adapter_scaffold_module_executes_with_python_m(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "custom_probe.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.adapter_scaffold",
            "--backend",
            "sglang",
            "--output-file",
            str(output_path),
            "--module-name",
            "custom_probe",
        ],
        check=True,
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
    assert completed.stderr == ""
    assert "native_probe_runtime_contract_to_record(\"sglang\")" in output_path.read_text(encoding="utf-8")
