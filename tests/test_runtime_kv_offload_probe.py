import json

from document_kv_cache.runtime_kv_offload_probe import (
    RUNTIME_KV_OFFLOAD_PROBE_RECORD_TYPE,
    run_runtime_kv_offload_probe,
    runtime_kv_offload_probe_record_issues,
    write_runtime_kv_offload_probe_json,
)


def test_runtime_kv_offload_probe_validates_config_and_hierarchy(tmp_path):
    record = run_runtime_kv_offload_probe(work_dir=tmp_path / "work")

    assert record["ok"] is True
    assert runtime_kv_offload_probe_record_issues(record) == ()
    assert record["record_type"] == RUNTIME_KV_OFFLOAD_PROBE_RECORD_TYPE
    assert record["vllm_runtime_kv_offload"]["config"]["kv_connector"] == "OffloadingConnector"
    assert record["vllm_runtime_kv_offload"]["config"]["kv_connector_extra_config"]["spec_name"] == (
        "TieringOffloadingSpec"
    )
    assert record["sglang_hicache"]["server_args"][0] == "--enable-hierarchical-cache"
    assert record["hierarchical_document_kv"]["disk_hot_tiers"] == [
        "cold_storage",
        "cold_storage",
        "local_disk",
    ]
    assert record["hierarchical_document_kv"]["memory_hot_tiers"] == [
        "cold_storage",
        "cold_storage",
        "local_disk",
    ]
    assert record["hierarchical_document_kv"]["stats"]["local_promotions"] >= 4
    assert record["hierarchical_document_kv"]["stats"]["local_evictions"] >= 3
    assert record["hierarchical_document_kv"]["cpu_to_local_promotion"]["stats"]["cpu_hits"] == 1


def test_write_runtime_kv_offload_probe_json(tmp_path):
    output_path = tmp_path / "probe.json"

    record = write_runtime_kv_offload_probe_json(output_path, work_dir=tmp_path / "work")

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == record
    assert loaded["ok"] is True
