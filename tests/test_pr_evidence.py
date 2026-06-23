import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import document_kv_cache.pr_evidence as public_pr_evidence
from document_kv_cache.pr_evidence import (
    GPT55_REVIEW_OUTCOMES,
    PR_EVIDENCE_RECORD_TYPE,
    PR_EVIDENCE_VALIDATION_RECORD_TYPE,
    PullRequestEvidence,
    evaluate_pr_evidence,
    evaluate_pr_evidence_directory,
    evaluate_pr_evidence_file,
    evaluate_pr_evidence_record,
    _github_pull_request_url_identity,
    pr_evidence_validation_to_record,
    pr_evidence_to_record,
    write_pr_evidence_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_pr_evidence_accepts_complete_traceability_record():
    evidence = evaluate_pr_evidence(
        pull_request_number=123,
        pull_request_url="https://github.com/puyuanOT/cachet/pull/123",
        what_changed=("Added release evidence latency guards",),
        why="Make the release gate reject impossible latency artifacts.",
        scope=("release-evidence", "tests"),
        verification=("poetry run pytest -q tests/test_release_evidence.py",),
        refactor_skill_applied=True,
        gpt55_review_completed=True,
        gpt55_review_findings_resolved=True,
        gpt55_review_outcome="clean",
        gpt55_review_summary="GPT-5.5 review returned no findings.",
    )
    record = pr_evidence_to_record(evidence)

    assert evidence.ok
    assert record == {
        "record_type": PR_EVIDENCE_RECORD_TYPE,
        "ok": True,
        "pull_request_number": 123,
        "pull_request_url": "https://github.com/puyuanOT/cachet/pull/123",
        "what_changed": ["Added release evidence latency guards"],
        "why": "Make the release gate reject impossible latency artifacts.",
        "scope": ["release-evidence", "tests"],
        "verification": ["poetry run pytest -q tests/test_release_evidence.py"],
        "refactor_skill_applied": True,
        "gpt55_review_completed": True,
        "gpt55_review_findings_resolved": True,
        "gpt55_review_outcome": "clean",
        "gpt55_review_summary": "GPT-5.5 review returned no findings.",
        "issues": [],
    }


def test_evaluate_pr_evidence_reports_missing_traceability_and_review_gates():
    evidence = evaluate_pr_evidence(
        what_changed=(),
        why=" ",
        scope=(),
        verification=(),
        refactor_skill_applied=False,
        gpt55_review_completed=False,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="",
        gpt55_review_summary="",
    )

    assert not evidence.ok
    assert evidence.issues == (
        "what_changed must include at least one item",
        "why must be non-empty",
        "scope must include at least one touched boundary",
        "verification must include tests, builds, benchmarks, or an explicit not-applicable note",
        "Refactor skill must be applied during the PR slice",
        "GPT-5.5 review must be completed",
        "gpt55_review_outcome must be 'clean' or 'findings_resolved'",
        "gpt55_review_summary must describe findings and fixes, or state that the review was clean",
    )


def test_pr_evidence_tracks_optional_pull_request_identity():
    evidence = evaluate_pr_evidence(
        pull_request_number=264,
        pull_request_url="https://github.com/puyuanOT/cachet/pull/264",
        what_changed=("Added Databricks status target validation",),
        why="Tie release evidence to the PR that changed the validator.",
        scope=("databricks_runs.py",),
        verification=("pytest -q tests/test_databricks_runs.py",),
        refactor_skill_applied=True,
        gpt55_review_completed=True,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="clean",
        gpt55_review_summary="Review was clean.",
    )

    assert evidence.ok
    assert pr_evidence_to_record(evidence)["pull_request_number"] == 264

    mismatched = evaluate_pr_evidence_record(
        {
            **pr_evidence_to_record(evidence),
            "pull_request_url": "https://github.com/puyuanOT/cachet/pull/999",
        }
    )

    assert not mismatched.ok
    assert "pull_request_url must end with pull_request_number when both are provided" in mismatched.issues


def test_pr_evidence_rejects_malformed_github_pull_request_urls():
    evidence = evaluate_pr_evidence(
        pull_request_number=123,
        pull_request_url="https://github.com/owner/cachet/pull/123",
        what_changed=("Hardened PR URL validation",),
        why="Release traceability must identify the exact GitHub pull request.",
        scope=("pr_evidence.py",),
        verification=("pytest -q tests/test_pr_evidence.py",),
        refactor_skill_applied=True,
        gpt55_review_completed=True,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="clean",
        gpt55_review_summary="Review was clean.",
    )
    assert _github_pull_request_url_identity(evidence.pull_request_url) == ("owner/cachet", 123)

    for malformed_url in (
        "https://github.com/pull/123",
        "https://github.com/owner/cachet/issues/123",
        "https://github.com/owner/cachet/pull/0",
        "https://github.com/owner/cachet/pull/123?debug=true",
        "https://github.com/owner/cachet?debug=true/pull/123",
        "https://github.com/owner/cachet#fragment/pull/123",
        "https://github.com/owner/document%20kv-cache/pull/123",
        "https://github.com/owner/cachet/pull/１２３",
        "https://github.com/owner/cachet/pull/١٢٣",
        "https://github.com/owner/cachet/pull/0123",
        "https://github.com//owner/cachet/pull/123",
        "https://github.com/owner/cachet/pull/123/",
    ):
        parsed = evaluate_pr_evidence_record(
            {
                **pr_evidence_to_record(evidence),
                "pull_request_url": malformed_url,
            }
        )

        assert not parsed.ok
        assert "pull_request_url must be a GitHub pull request URL when provided" in parsed.issues


def test_pr_evidence_dataclass_validates_json_safe_schema_and_semantics():
    evidence = PullRequestEvidence(
        what_changed=[" changed "],
        why=" why ",
        scope=[" docs "],
        verification=[" tests "],
        refactor_skill_applied=True,
        gpt55_review_completed=True,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="clean",
        gpt55_review_summary=" clean ",
    )

    assert evidence.what_changed == ("changed",)
    assert evidence.gpt55_review_summary == "clean"
    assert evidence.ok

    with pytest.raises(TypeError, match="what_changed"):
        PullRequestEvidence(
            what_changed="changed",
            why="why",
            scope=("docs",),
            verification=("tests",),
            refactor_skill_applied=True,
            gpt55_review_completed=True,
            gpt55_review_findings_resolved=True,
            gpt55_review_outcome="clean",
            gpt55_review_summary="clean",
        )
    with pytest.raises(ValueError, match="refactor_skill_applied"):
        PullRequestEvidence(
            what_changed=("changed",),
            why="why",
            scope=("docs",),
            verification=("tests",),
            refactor_skill_applied=1,
            gpt55_review_completed=True,
            gpt55_review_findings_resolved=True,
            gpt55_review_outcome="clean",
            gpt55_review_summary="clean",
        )


def test_pr_evidence_dataclass_derives_issues_for_invalid_direct_objects():
    evidence = PullRequestEvidence(
        what_changed=(),
        why="",
        scope=(),
        verification=(),
        refactor_skill_applied=False,
        gpt55_review_completed=False,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="findings_resolved",
        gpt55_review_summary="",
    )

    assert not evidence.ok
    assert "what_changed must include at least one item" in evidence.issues
    assert "GPT-5.5 findings must be resolved when gpt55_review_outcome is 'findings_resolved'" in evidence.issues


def test_pr_evidence_dataclass_preserves_positional_issues_argument():
    evidence = PullRequestEvidence(
        ("changed",),
        "why",
        ("scope",),
        ("pytest",),
        True,
        True,
        False,
        "clean",
        "review was clean",
        ("caller supplied issue",),
    )

    assert "caller supplied issue" in evidence.issues
    assert evidence.pull_request_number is None
    assert evidence.pull_request_url == ""


def test_evaluate_pr_evidence_record_rejects_wrong_record_type():
    evidence = evaluate_pr_evidence_record({"record_type": "document_kv.release_evidence.v1"})

    assert not evidence.ok
    assert f"record_type must be {PR_EVIDENCE_RECORD_TYPE!r}" in evidence.issues


def test_evaluate_pr_evidence_record_rejects_unsupported_keys():
    evidence = evaluate_pr_evidence(
        what_changed=("Tightened PR evidence parsing",),
        why="Keep PR traceability records auditable.",
        scope=("pr_evidence.py",),
        verification=("poetry run pytest tests/test_pr_evidence.py",),
        refactor_skill_applied=True,
        gpt55_review_completed=True,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="clean",
        gpt55_review_summary="Review was clean.",
    )
    record = pr_evidence_to_record(evidence)
    record["debug"] = {"accepted": False}

    parsed = evaluate_pr_evidence_record(record)

    assert not parsed.ok
    assert "PR evidence record has unsupported keys: ['debug']" in parsed.issues


def test_evaluate_pr_evidence_record_returns_failed_evidence_for_malformed_current_records():
    evidence = evaluate_pr_evidence_record(
        {
            "record_type": PR_EVIDENCE_RECORD_TYPE,
            "pull_request_number": 0,
            "pull_request_url": "https://example.com/pull/1",
            "what_changed": ["ok", 3],
            "why": None,
            "scope": "governance",
            "verification": ["tests"],
            "refactor_skill_applied": "true",
            "gpt55_review_completed": True,
            "gpt55_review_findings_resolved": False,
            "gpt55_review_outcome": "findings_resolved",
            "gpt55_review_summary": 42,
        }
    )

    assert not evidence.ok
    assert "what_changed[1] must be a non-empty string" in evidence.issues
    assert "pull_request_number must be a positive integer or null" in evidence.issues
    assert "pull_request_url must be a GitHub pull request URL when provided" in evidence.issues
    assert "why must be a string" in evidence.issues
    assert "scope must be a sequence" in evidence.issues
    assert "refactor_skill_applied must be boolean" in evidence.issues
    assert "gpt55_review_summary must be a string" in evidence.issues
    assert "GPT-5.5 findings must be resolved when gpt55_review_outcome is 'findings_resolved'" in evidence.issues


def test_evaluate_pr_evidence_file_reports_invalid_json_and_non_object_records(tmp_path):
    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"\xff")
    invalid_utf8_evidence = evaluate_pr_evidence_file(invalid_utf8)

    assert not invalid_utf8_evidence.ok
    assert any("must contain valid UTF-8 JSON" in issue for issue in invalid_utf8_evidence.issues)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    invalid_json_evidence = evaluate_pr_evidence_file(invalid_json)

    assert not invalid_json_evidence.ok
    assert any("must contain valid JSON" in issue for issue in invalid_json_evidence.issues)

    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    array_evidence = evaluate_pr_evidence_file(array_json)

    assert not array_evidence.ok
    assert any("must contain a JSON object" in issue for issue in array_evidence.issues)


def test_evaluate_pr_evidence_directory_sorts_and_validates_sidecars(tmp_path):
    valid = evaluate_pr_evidence(
        what_changed=("Added PR evidence directory validation",),
        why="Make repository traceability evidence machine-checkable.",
        scope=("governance",),
        verification=("poetry run pytest -q tests/test_pr_evidence.py",),
        refactor_skill_applied=True,
        gpt55_review_completed=True,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="clean",
        gpt55_review_summary="Review was clean.",
    )
    write_pr_evidence_json(valid, tmp_path / "b-valid.json")
    write_pr_evidence_json(valid, tmp_path / "nested" / "c-valid.json")
    validation_record = pr_evidence_validation_to_record({"b-valid.json": valid})
    (tmp_path / "nested" / "pr-evidence-validation.json").write_text(
        json.dumps(validation_record),
        encoding="utf-8",
    )
    raw_validation_record = dict(validation_record)
    raw_validation_record["debug"] = {"accepted": False}
    (tmp_path / "nested" / "raw-validation-summary.json").write_text(
        json.dumps(raw_validation_record),
        encoding="utf-8",
    )
    (tmp_path / "nested" / "bad-validation-summary.json").write_text(
        json.dumps({"record_type": PR_EVIDENCE_VALIDATION_RECORD_TYPE}),
        encoding="utf-8",
    )
    (tmp_path / "nested" / "empty-validation-summary.json").write_text(
        json.dumps(
            {
                "record_type": PR_EVIDENCE_VALIDATION_RECORD_TYPE,
                "ok": True,
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "a-invalid.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nested" / "d-invalid.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("not a sidecar", encoding="utf-8")

    evidence_by_file = evaluate_pr_evidence_directory(tmp_path)

    assert list(evidence_by_file) == [
        "a-invalid.json",
        "b-valid.json",
        "nested/bad-validation-summary.json",
        "nested/c-valid.json",
        "nested/d-invalid.json",
        "nested/empty-validation-summary.json",
        "nested/raw-validation-summary.json",
    ]
    assert not evidence_by_file["a-invalid.json"].ok
    assert evidence_by_file["b-valid.json"].ok
    assert not evidence_by_file["nested/bad-validation-summary.json"].ok
    assert any(
        "record_type must be 'document_kv.pr_evidence.v1'" in issue
        for issue in evidence_by_file["nested/bad-validation-summary.json"].issues
    )
    assert evidence_by_file["nested/c-valid.json"].ok
    assert not evidence_by_file["nested/d-invalid.json"].ok
    assert not evidence_by_file["nested/empty-validation-summary.json"].ok
    assert not evidence_by_file["nested/raw-validation-summary.json"].ok

    shallow_evidence_by_file = evaluate_pr_evidence_directory(tmp_path, recursive=False)

    assert list(shallow_evidence_by_file) == ["a-invalid.json", "b-valid.json"]


def test_pr_evidence_validation_to_record_summarizes_per_file_evidence(tmp_path):
    valid = evaluate_pr_evidence(
        what_changed=("Added PR evidence validation summary",),
        why="Make CLI validation machine-readable.",
        scope=("governance",),
        verification=("poetry run pytest -q tests/test_pr_evidence.py",),
        refactor_skill_applied=True,
        gpt55_review_completed=True,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="clean",
        gpt55_review_summary="Review was clean.",
    )
    invalid = evaluate_pr_evidence_record({})

    record = pr_evidence_validation_to_record({"invalid.json": invalid, "valid.json": valid})

    assert record["record_type"] == PR_EVIDENCE_VALIDATION_RECORD_TYPE
    assert record["ok"] is False
    assert record["files"]["valid.json"]["ok"] is True
    assert record["files"]["invalid.json"]["ok"] is False
    assert pr_evidence_validation_to_record({})["ok"] is False


def test_repository_pr_evidence_sidecars_are_valid():
    evidence_by_file = evaluate_pr_evidence_directory(REPO_ROOT / "pr-evidence")

    assert evidence_by_file
    assert {path: evidence.issues for path, evidence in evidence_by_file.items() if not evidence.ok} == {}


def test_write_pr_evidence_json_round_trips(tmp_path):
    output_path = tmp_path / "nested" / "pr-evidence.json"
    evidence = evaluate_pr_evidence(
        what_changed=("Updated PR evidence docs",),
        why="Keep traceability machine-checkable.",
        scope=("governance",),
        verification=("poetry run pytest -q tests/test_pr_evidence.py",),
        refactor_skill_applied=True,
        gpt55_review_completed=True,
        gpt55_review_findings_resolved=True,
        gpt55_review_outcome="findings_resolved",
        gpt55_review_summary="Review findings were fixed.",
    )

    write_pr_evidence_json(evidence, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == pr_evidence_to_record(evidence)


def test_public_pr_evidence_cli_uses_wrapper_hooks(monkeypatch, tmp_path):
    output_path = tmp_path / "pr-evidence.json"
    called = {}

    def fake_evaluate_pr_evidence(**kwargs):
        called["kwargs"] = kwargs
        return PullRequestEvidence(
            what_changed=("public wrapper hook",),
            why="prove hook bridging",
            scope=("governance",),
            verification=("hooked",),
            refactor_skill_applied=True,
            gpt55_review_completed=True,
            gpt55_review_findings_resolved=True,
            gpt55_review_outcome="clean",
            gpt55_review_summary="clean",
        )

    monkeypatch.setattr(public_pr_evidence, "evaluate_pr_evidence", fake_evaluate_pr_evidence)

    exit_code = public_pr_evidence.main(
        [
            "--what-changed",
            "ignored",
            "--why",
            "ignored",
            "--scope",
            "ignored",
            "--verification",
            "ignored",
            "--pull-request-number",
            "321",
            "--pull-request-url",
            "https://github.com/puyuanOT/cachet/pull/321",
            "--refactor-skill-applied",
            "--gpt55-review-completed",
            "--gpt55-review-findings-resolved",
            "--gpt55-review-outcome",
            "findings_resolved",
            "--gpt55-review-summary",
            "ignored",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert called["kwargs"]["what_changed"] == ("ignored",)
    assert called["kwargs"]["pull_request_number"] == 321
    assert called["kwargs"]["pull_request_url"] == "https://github.com/puyuanOT/cachet/pull/321"
    assert called["kwargs"]["gpt55_review_outcome"] == "findings_resolved"
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["what_changed"] == ["public wrapper hook"]


def test_public_pr_evidence_cli_validates_directory_with_wrapper_hooks(monkeypatch, tmp_path):
    output_path = tmp_path / "validation.json"
    called = {}

    def fake_evaluate_pr_evidence_directory(directory, *, pattern="*.json"):
        called["directory"] = directory
        called["pattern"] = pattern
        return {
            "valid.json": PullRequestEvidence(
                what_changed=("public validation hook",),
                why="prove validation hook bridging",
                scope=("governance",),
                verification=("hooked",),
                refactor_skill_applied=True,
                gpt55_review_completed=True,
                gpt55_review_findings_resolved=False,
                gpt55_review_outcome="clean",
                gpt55_review_summary="clean",
            )
        }

    monkeypatch.setattr(public_pr_evidence, "evaluate_pr_evidence_directory", fake_evaluate_pr_evidence_directory)

    exit_code = public_pr_evidence.main(
        [
            "--validate-directory",
            str(tmp_path),
            "--output-json",
            str(output_path),
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert called == {"directory": str(tmp_path), "pattern": "*.json"}
    assert record["record_type"] == PR_EVIDENCE_VALIDATION_RECORD_TYPE
    assert record["files"][f"{tmp_path}/valid.json"]["what_changed"] == ["public validation hook"]


def test_public_pr_evidence_cli_validates_json_with_record_wrapper_hook(monkeypatch, tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "validation.json"
    called = {}

    def fake_evaluate_pr_evidence_record(record):
        called["record"] = record
        return PullRequestEvidence(
            what_changed=("public record validation hook",),
            why="prove record hook bridging",
            scope=("governance",),
            verification=("hooked",),
            refactor_skill_applied=True,
            gpt55_review_completed=True,
            gpt55_review_findings_resolved=False,
            gpt55_review_outcome="clean",
            gpt55_review_summary="clean",
        )

    monkeypatch.setattr(public_pr_evidence, "evaluate_pr_evidence_record", fake_evaluate_pr_evidence_record)

    exit_code = public_pr_evidence.main(
        [
            "--validate-json",
            str(input_path),
            "--output-json",
            str(output_path),
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert called == {"record": {}}
    assert record["files"][str(input_path)]["what_changed"] == ["public record validation hook"]


def test_pr_evidence_cli_writes_failed_evidence_for_missing_gates(tmp_path):
    output_path = tmp_path / "failed-pr-evidence.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.pr_evidence",
            "--output-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert completed.returncode == 2
    assert record["ok"] is False
    assert "Refactor skill must be applied during the PR slice" in record["issues"]
    assert "gpt55_review_outcome must be 'clean' or 'findings_resolved'" in record["issues"]
    assert GPT55_REVIEW_OUTCOMES == ("clean", "findings_resolved")


def test_pr_evidence_cli_validates_directory_and_sets_exit_code(tmp_path):
    valid = evaluate_pr_evidence(
        what_changed=("Added CLI directory validation",),
        why="Let operators validate evidence before bundling.",
        scope=("governance",),
        verification=("poetry run pytest -q tests/test_pr_evidence.py",),
        refactor_skill_applied=True,
        gpt55_review_completed=True,
        gpt55_review_findings_resolved=False,
        gpt55_review_outcome="clean",
        gpt55_review_summary="Review was clean.",
    )
    write_pr_evidence_json(valid, tmp_path / "valid.json")
    (tmp_path / "invalid.json").write_text("{}", encoding="utf-8")
    output_path = tmp_path / "validation.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.pr_evidence",
            "--validate-directory",
            str(tmp_path),
            "--output-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert completed.returncode == 2
    assert record["record_type"] == PR_EVIDENCE_VALIDATION_RECORD_TYPE
    assert record["ok"] is False
    assert record["files"][f"{tmp_path}/valid.json"]["ok"] is True
    assert record["files"][f"{tmp_path}/invalid.json"]["ok"] is False
