"""Original Jobs GET envelopes keep observed bytes separate from proved identity."""

import copy
import hashlib
import io
import json

import pytest

from document_kv_cache.databricks_runs import (
    DatabricksWorkspaceConfig,
    _validated_original_attempt_run_id,
    canonical_databricks_submit_payload_snapshot,
    get_databricks_run,
)


@pytest.fixture
def original_run():
    # Sanitized shape of the actual original API 2.1 SUBMIT_RUN response:
    # job_run_id is present; parent attempt/original/repair fields are absent.
    return {
        "run_id": 701,
        "job_run_id": 701,
        "run_type": "SUBMIT_RUN",
        "run_name": "receipt-bound-original",
        "start_time": 1000,
        "end_time": 3000,
        "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        "tasks": [
            {
                "run_id": 1701,
                "task_key": "worker",
                "attempt_number": 0,
                "start_time": 1100,
                "end_time": 2900,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
            }
        ],
    }


def test_direct_get_omission_proves_receipt_identity_without_rewriting_raw(
    original_run,
):
    wire_bytes = (json.dumps(original_run, indent=2) + "\n").encode()

    class Response(io.BytesIO):
        status = 200

    def opener(request, *, timeout):
        assert request.method == "GET"
        assert request.full_url.endswith("/api/2.1/jobs/runs/get?run_id=701")
        assert timeout > 0
        return Response(wire_bytes)

    run = get_databricks_run(
        DatabricksWorkspaceConfig("https://workspace.example", "test-token"),
        "701",
        opener=opener,
    )
    before = copy.deepcopy(run)
    _, canonical_before = canonical_databricks_submit_payload_snapshot(run)
    assert _validated_original_attempt_run_id(run, expected_run_id="701") == "701"
    _, canonical_after = canonical_databricks_submit_payload_snapshot(run)
    assert run == before == original_run
    assert "original_attempt_run_id" not in run
    assert canonical_after == canonical_before
    assert (
        hashlib.sha256(canonical_after).digest()
        == hashlib.sha256(canonical_before).digest()
    )


@pytest.mark.parametrize(
    "aliases",
    [
        {},
        {"original_attempt_run_id": 701},
        {"original_attempt_run_id": "701", "job_run_id": 701},
        {"original_attempt_run_id": 701, "job_run_id": "701"},
        {"job_run_id": 701, "attempt_number": 0, "repair_history": []},
    ],
)
def test_matching_optional_identity_aliases_are_corroboration(original_run, aliases):
    original_run.pop("job_run_id")
    original_run.update(aliases)
    before = copy.deepcopy(original_run)
    assert (
        _validated_original_attempt_run_id(original_run, expected_run_id=701) == "701"
    )
    assert original_run == before


@pytest.mark.parametrize("field", ["original_attempt_run_id", "job_run_id"])
@pytest.mark.parametrize(
    "value", [None, 0, "0", False, True, 701.0, "0701", " 701", 702]
)
def test_present_invalid_or_conflicting_alias_is_never_treated_as_absent(
    original_run, field, value
):
    original_run[field] = value
    before = copy.deepcopy(original_run)
    with pytest.raises(ValueError, match=field):
        _validated_original_attempt_run_id(original_run, expected_run_id="701")
    assert original_run == before


@pytest.mark.parametrize("value", [None, False, True, 0.0, "0", 1, -1])
@pytest.mark.parametrize("level", ["parent", "task"])
def test_attempt_number_requires_exact_integer_zero(original_run, level, value):
    target = original_run if level == "parent" else original_run["tasks"][0]
    target["attempt_number"] = value
    with pytest.raises(ValueError, match="attempt zero"):
        _validated_original_attempt_run_id(original_run, expected_run_id="701")


@pytest.mark.parametrize("tasks", [None, [], [{}, {}], [{}], [None]])
def test_missing_or_ambiguous_task_attempt_cannot_prove_original(original_run, tasks):
    original_run["tasks"] = tasks
    with pytest.raises(ValueError, match="one task attempt zero"):
        _validated_original_attempt_run_id(original_run, expected_run_id="701")


@pytest.mark.parametrize("repairs", [[{"type": "REPAIR_ALL"}], {}, False, ""])
def test_repaired_or_malformed_history_cannot_prove_original(original_run, repairs):
    original_run["repair_history"] = repairs
    with pytest.raises(ValueError, match="repair history"):
        _validated_original_attempt_run_id(original_run, expected_run_id="701")


@pytest.mark.parametrize("parent", [None, False, 0, "0", "0701", "701 ", 702])
def test_raw_parent_must_match_the_independent_receipt(original_run, parent):
    original_run["run_id"] = parent
    with pytest.raises(ValueError, match="terminal run_id"):
        _validated_original_attempt_run_id(original_run, expected_run_id="701")


@pytest.mark.parametrize("receipt", [None, False, 0, "0", "0701", "701 ", 702])
def test_invalid_or_different_receipt_cannot_be_replaced_by_observed_parent(
    original_run, receipt
):
    with pytest.raises(ValueError, match="receipt"):
        _validated_original_attempt_run_id(original_run, expected_run_id=receipt)
