"""CPU contract and native-path tests; these fixtures are never GPU evidence."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import document_kv_cache.representative_runtime_qualification as qualification


def bindings():
    return {
        **{name: "1" * 64 for name in qualification._SHA_FIELDS},
        "source_commit": "2" * 40,
        "benchmark_id": "synthetic-unit-test",
        "runtime_id": "synthetic-unit-test-runtime",
        "arm_id": "document_kv_cache:full_prefix_prefill",
        "context_tokens": 8192,
    }


def unit_record():
    # Deliberately synthetic CPU validator fixture; never written as evidence.
    cases = []
    for encoding in ("stored_post_rope", "pre_rope"):
        cases.append(
            {
                "key_position_encoding": encoding,
                "persisted_payload_sha256": "3" * 64,
                "loaded_payload_sha256": "3" * 64,
                "persisted_payload_bytes": 4_866_048,
                "loaded_layers": 36,
                "injected_layers": 36,
                "segment_count": 2 if encoding == "pre_rope" else 1,
                "provider_load_calls": 2 if encoding == "pre_rope" else 1,
                "value_mismatch_count": 0,
                "unused_slot_mismatch_count": 0,
                "key_max_abs_error": 0.0,
                "attention_max_abs_error_by_layer": [0.0] * 36,
                "attention_launches": 36,
                "unrotated_key_negative_control_max_abs_delta": 2.0,
                "local_position_reset_negative_control_max_abs_delta": 2.0,
            }
        )
    record = {
        "record_type": qualification.RECORD_TYPE,
        "schema_version": 1,
        "scope": qualification.QUALIFICATION_SCOPE,
        "bindings": bindings(),
        "geometry": deepcopy(qualification._GEOMETRY),
        "runtime": {
            "python": "3.11.16",
            "torch": "2.13.0+cu129",
            "vllm": "0.27.1+cu129",
            "triton": "3.7.1",
            "cuda": "12.9",
            "device_type": "cuda",
            "device_count": 1,
            "device_name": "NVIDIA L4",
            "device_capability": [8, 9],
            "bf16_supported": True,
            "provider_source_sha256": "4" * 64,
            "attention_source_sha256": "5" * 64,
        },
        "cases": cases,
        "started_epoch_ns": 1,
        "finished_epoch_ns": 2,
        "model_forward_executed": False,
        "full_context_semantic_parity_claimed": False,
        "closed_record_sha256": "",
    }
    record["closed_record_sha256"] = qualification._closed_hash(record)
    return record


def reclose(record):
    record["closed_record_sha256"] = qualification._closed_hash(record)
    return record


def test_import_does_not_import_torch_or_touch_gpu():
    source_root = str(Path(qualification.__file__).resolve().parents[1])
    code = (
        "import sys; "
        f"sys.path.insert(0, {source_root!r}); "
        "import document_kv_cache.representative_runtime_qualification; "
        "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-B", "-S", "-c", code], check=True)


def test_closed_validator_is_pure_and_returns_a_detached_record():
    record = unit_record()
    validated = qualification.validate_representative_runtime_qualification_record(
        record, expected_bindings=bindings()
    )
    assert validated == record
    validated["cases"][0]["attention_max_abs_error_by_layer"][0] = 99
    assert record["cases"][0]["attention_max_abs_error_by_layer"][0] == 0.0


@pytest.mark.parametrize("field", tuple(bindings()))
def test_each_task_source_and_staging_binding_is_compared(field):
    record = unit_record()
    if field == "context_tokens":
        record["bindings"][field] = 16384
    elif field == "arm_id":
        record["bindings"][field] = "baseline_prefill"
    elif field == "source_commit":
        record["bindings"][field] = "a" * 40
    else:
        record["bindings"][field] = "a" * 64
    with pytest.raises(ValueError, match="binding drift"):
        qualification.validate_representative_runtime_qualification_record(
            reclose(record), expected_bindings=bindings()
        )


@pytest.mark.parametrize(
    "change",
    (
        {"device_type": "cpu"},
        {"device_count": 0},
        {"device_count": True},
        {"device_name": "NVIDIA L40S"},
        {"device_capability": [8, 0]},
        {"bf16_supported": False},
        {"vllm": "0.27.1"},
        {"python": "3.12.14"},
        {"torch": "2.12.1"},
        {"triton": "other"},
        {"cuda": None},
        {"provider_source_sha256": "not-a-digest"},
    ),
)
def test_validator_rejects_import_only_cpu_or_wrong_runtime_claims(change):
    record = unit_record()
    record["runtime"].update(change)
    with pytest.raises(ValueError):
        qualification.validate_representative_runtime_qualification_record(
            reclose(record), expected_bindings=bindings()
        )


@pytest.mark.parametrize(
    "change",
    (
        {"loaded_layers": 35},
        {"injected_layers": 35},
        {"attention_launches": 35},
        {"provider_load_calls": 0},
        {"loaded_payload_sha256": "f" * 64},
        {"value_mismatch_count": 1},
        {"unused_slot_mismatch_count": 1},
        {"key_max_abs_error": 0.1},
        {"attention_max_abs_error_by_layer": [0.0] * 35},
        {"attention_max_abs_error_by_layer": [0.0] * 35 + [0.1]},
        {"unrotated_key_negative_control_max_abs_delta": 0.0},
        {"local_position_reset_negative_control_max_abs_delta": 0.0},
    ),
)
def test_validator_rejects_incomplete_or_failed_real_checks_even_when_reclosed(change):
    record = unit_record()
    record["cases"][0].update(change)
    with pytest.raises(ValueError):
        qualification.validate_representative_runtime_qualification_record(
            reclose(record), expected_bindings=bindings()
        )


@pytest.mark.parametrize(
    "change",
    (
        {"scope": "model-logit-equivalence"},
        {"model_forward_executed": True},
        {"full_context_semantic_parity_claimed": True},
        {"cases": []},
        {"finished_epoch_ns": 1},
        {"started_epoch_ns": False},
        {"unexpected": True},
    ),
)
def test_validator_rejects_scope_inflation_missing_coverage_and_invalid_closure(change):
    record = unit_record()
    record.update(change)
    with pytest.raises(ValueError):
        qualification.validate_representative_runtime_qualification_record(
            reclose(record), expected_bindings=bindings()
        )
    record = unit_record()
    record["closed_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="closure drift"):
        qualification.validate_representative_runtime_qualification_record(
            record, expected_bindings=bindings()
        )


def test_real_entrypoint_cannot_emit_success_on_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "qualification.json"
    with pytest.raises(RuntimeError, match="real CUDA BF16"):
        qualification.run_representative_runtime_qualification(
            output_path=output, expected_bindings=bindings()
        )
    assert list(tmp_path.iterdir()) == []


def test_fresh_output_and_real_directory_are_required_before_gpu_access(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("untouched")
    with pytest.raises(ValueError, match="fresh"):
        qualification.run_representative_runtime_qualification(
            output_path=output, expected_bindings=bindings()
        )
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        qualification.run_representative_runtime_qualification(
            output_path=link / "new.json", expected_bindings=bindings()
        )
    assert output.read_text() == "untouched"


@pytest.mark.parametrize("pre_rope", (False, True))
def test_actual_cpu_provider_roundtrip_covers_all_36_layers_pages_and_global_rope(
    tmp_path, pre_rope
):
    torch = pytest.importorskip("torch")
    case, pages, keys, values = qualification._native_roundtrip(
        torch, tmp_path, pre_rope=pre_rope, device="cpu"
    )
    assert len(pages) == keys.shape[1] == values.shape[1] == 36
    assert case["persisted_payload_bytes"] == 4_866_048
    assert case["persisted_payload_sha256"] == case["loaded_payload_sha256"]
    assert case["value_mismatch_count"] == case["unused_slot_mismatch_count"] == 0
    assert case["key_max_abs_error"] <= (1 / 128 if pre_rope else 0)
    assert case["local_position_reset_negative_control_max_abs_delta"] > 0.25
    assert case["unrotated_key_negative_control_max_abs_delta"] > 0.25
    assert not list(tmp_path.glob("*.json"))  # No fabricated GPU result.
    case_root = tmp_path / ("pre_rope" if pre_rope else "stored_post_rope")
    handoff = json.loads((case_root / "handoff.json").read_text())
    assert (
        handoff["handle"]["layout"]["key_position_encoding"]
        == case["key_position_encoding"]
    )


def test_actual_native_pre_rope_position_reset_bug_is_detected(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    import vllm_kv_injection.vllm_native_provider as native

    original = native._rope_cos_sin_for_load

    def wrong_local_positions(load, *args, **kwargs):
        local_load = SimpleNamespace(source_token_start=0, token_count=load.token_count)
        return original(local_load, *args, **kwargs)

    monkeypatch.setattr(native, "_rope_cos_sin_for_load", wrong_local_positions)
    case, *_ = qualification._native_roundtrip(
        torch, tmp_path, pre_rope=True, device="cpu"
    )
    assert case["key_max_abs_error"] > 0.25
    case.update(attention_max_abs_error_by_layer=[0.0] * 36, attention_launches=36)
    with pytest.raises(ValueError, match="key_max_abs_error"):
        qualification._validate_case(case, "pre_rope")


def test_reference_gqa_zero_query_equals_each_kv_heads_value_mean():
    torch = pytest.importorskip("torch")
    query = torch.zeros((1, 32, 128), dtype=torch.bfloat16)
    keys = torch.ones((33, 8, 128), dtype=torch.bfloat16)
    values = torch.arange(8, dtype=torch.bfloat16)[None, :, None].expand(33, 8, 128)
    actual = qualification._attention_reference(torch, query, keys, values)
    expected = (
        torch.arange(8, dtype=torch.float32)
        .repeat_interleave(4)[None, :, None]
        .expand(1, 32, 128)
    )
    assert torch.allclose(actual, expected)


def test_cli_forwards_exact_binding_file_and_output(tmp_path, monkeypatch):
    path = tmp_path / "bindings.json"
    path.write_text(json.dumps(bindings()))
    output = tmp_path / "qualification.json"
    calls = []
    monkeypatch.setattr(
        qualification,
        "run_representative_runtime_qualification",
        lambda **kwargs: calls.append(kwargs),
    )
    assert (
        qualification.main(["--bindings-json", str(path), "--output-json", str(output)])
        == 0
    )
    assert calls == [{"output_path": output, "expected_bindings": bindings()}]
    assert not output.exists()
