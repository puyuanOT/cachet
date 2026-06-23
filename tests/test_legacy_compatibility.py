import json
import os
import subprocess
import sys
from pathlib import Path

from document_kv_cache.legacy_compatibility import (
    LEGACY_COMPATIBILITY_MIGRATION_RECORD_TYPE,
    LEGACY_COMPATIBILITY_MIGRATION_VALIDATION_RECORD_TYPE,
    LegacyCompatibilityMigrationEvidence,
    evaluate_legacy_compatibility_migration_file,
    evaluate_legacy_compatibility_migration_record,
    legacy_compatibility_migration_to_record,
    legacy_compatibility_migration_validation_to_record,
    write_legacy_compatibility_migration_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_compatibility_migration_evidence_accepts_complete_record():
    record = _migration_record()

    evidence = evaluate_legacy_compatibility_migration_record(record)

    assert evidence.ok is True
    assert evidence.issues == ()
    assert legacy_compatibility_migration_to_record(evidence) == record


def test_legacy_compatibility_migration_evidence_reports_missing_categories_and_legacy_usage():
    record = _migration_record()
    record["checked_downstream_jobs"] = record["checked_downstream_jobs"][:1]
    record["checked_downstream_jobs"][0]["legacy_imports_present"] = True
    record["checked_downstream_jobs"][0]["legacy_console_scripts_present"] = True
    record["release_evidence"] = [
        {
            "hardware_target": "aws-g5-a10g",
            "evidence_uri": "dbfs:/evidence/g5-release.json",
            "runner_uses_legacy_facade": True,
        }
    ]

    evidence = evaluate_legacy_compatibility_migration_record(record)

    assert evidence.ok is False
    assert "checked_downstream_jobs[0].legacy_imports_present must be false" in evidence.issues
    assert "checked_downstream_jobs[0].legacy_console_scripts_present must be false" in evidence.issues
    assert "checked_downstream_jobs must cover required categories: benchmark, storage, native_probe, smoke" in (
        evidence.issues
    )
    assert "release_evidence[0].runner_uses_legacy_facade must be false" in evidence.issues
    assert "release_evidence must include the strict V1 hardware target 'aws-g6-l4'" in evidence.issues


def test_legacy_compatibility_migration_evidence_rejects_wrong_schema_and_unsupported_keys():
    record = _migration_record()
    record["unexpected"] = True
    record["checked_downstream_jobs"][0]["unexpected"] = True
    record["release_evidence"][0]["unexpected"] = True

    evidence = evaluate_legacy_compatibility_migration_record(record)
    wrong_type = evaluate_legacy_compatibility_migration_record({"record_type": "document_kv.pr_evidence.v1"})

    assert evidence.ok is False
    assert "legacy migration evidence has unsupported keys: ['unexpected']" in evidence.issues
    assert "checked_downstream_jobs[0] has unsupported keys: ['unexpected']" in evidence.issues
    assert "release_evidence[0] has unsupported keys: ['unexpected']" in evidence.issues
    assert wrong_type.issues == (
        f"record_type must be {LEGACY_COMPATIBILITY_MIGRATION_RECORD_TYPE!r}",
    )


def test_legacy_compatibility_migration_evidence_file_and_writer_round_trip(tmp_path):
    evidence = evaluate_legacy_compatibility_migration_record(_migration_record())
    output_path = tmp_path / "legacy-migration.json"

    write_legacy_compatibility_migration_json(evidence, output_path)

    assert evaluate_legacy_compatibility_migration_file(output_path) == evidence
    assert json.loads(output_path.read_text(encoding="utf-8")) == _migration_record()


def test_legacy_compatibility_migration_validation_record_summarizes_files():
    valid = evaluate_legacy_compatibility_migration_record(_migration_record())
    invalid = evaluate_legacy_compatibility_migration_record({})

    record = legacy_compatibility_migration_validation_to_record(
        {
            "valid.json": valid,
            "invalid.json": invalid,
        }
    )

    assert record["record_type"] == LEGACY_COMPATIBILITY_MIGRATION_VALIDATION_RECORD_TYPE
    assert record["ok"] is False
    assert record["files"]["valid.json"]["ok"] is True
    assert record["files"]["invalid.json"]["ok"] is False


def test_legacy_compatibility_migration_cli_validates_json(tmp_path):
    evidence_path = tmp_path / "legacy-migration.json"
    evidence_path.write_text(json.dumps(_migration_record()), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.legacy_compatibility",
            "--validate-json",
            str(evidence_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    cachet_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cachet.legacy_compatibility",
            "--validate-json",
            str(evidence_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(completed.stdout)["ok"] is True
    assert json.loads(cachet_completed.stdout)["ok"] is True


def test_legacy_compatibility_migration_evidence_dataclass_rejects_bad_issue_values():
    try:
        LegacyCompatibilityMigrationEvidence(
            checked_downstream_jobs=(),
            release_evidence=(),
            issues=("ok", ""),
        )
    except ValueError as exc:
        assert "issues entries must be non-empty strings" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("empty issue values should be rejected")


def _migration_record():
    return {
        "record_type": LEGACY_COMPATIBILITY_MIGRATION_RECORD_TYPE,
        "ok": True,
        "checked_downstream_jobs": [
            _checked_job("release", "cachet-release-bundle"),
            _checked_job("benchmark", "cachet-vllm-benchmark"),
            _checked_job("storage", "cachet-storage-benchmark"),
            _checked_job("native_probe", "cachet-native-engine-probe"),
            _checked_job("smoke", "cachet-vllm-smoke"),
        ],
        "release_evidence": [
            {
                "hardware_target": "aws-g6-l4",
                "evidence_uri": "dbfs:/evidence/g6-release.json",
                "runner_uses_legacy_facade": False,
            },
            {
                "hardware_target": "aws-g5-a10g",
                "evidence_uri": "dbfs:/evidence/g5-compatibility.json",
                "runner_uses_legacy_facade": False,
            },
        ],
        "issues": [],
    }


def _checked_job(category: str, name: str):
    return {
        "name": name,
        "category": category,
        "environment": "Databricks QA",
        "migrated_import_surface": "cachet",
        "migrated_command_prefix": "cachet-",
        "legacy_imports_present": False,
        "legacy_console_scripts_present": False,
        "evidence_uri": f"dbfs:/migration/{name}.json",
    }
