"""Small public-API ledger histories for offline qualification unit tests."""

from pathlib import Path

import document_kv_cache.databricks_resource_ledger as ledger_api
import document_kv_cache.gpu_qualification as qualification_api
import document_kv_cache.publication_campaign as campaign_api


def bind_qualification_opening(monkeypatch, ledger):
    """Bind test-owned predecessor pins without mocking ledger validation.

    Campaign materialization is a separate unit boundary: use the real frozen
    campaign builder for that provenance, while the qualification controller's
    opening prefix and accounting are supplied by this synthetic history.
    """
    prefix = ledger_api.databricks_ledger_prefix(ledger)
    hours = ledger.terminal_actual_cluster_hours

    def campaign_builder(campaign_id, **kwargs):
        kwargs["campaign_ledger_prefix"] = (
            campaign_api.PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX
        )
        kwargs["campaign_opening_terminal_gpu_hours"] = (
            campaign_api.PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        )
        return campaign_api.build_publication_campaign_plan(campaign_id, **kwargs)

    monkeypatch.setattr(
        qualification_api, "build_publication_campaign_plan", campaign_builder
    )
    monkeypatch.setattr(
        qualification_api, "PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX", prefix
    )
    monkeypatch.setattr(
        qualification_api, "PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS", hours
    )
    return prefix, hours


def write_opening_ledger(
    path: Path,
    *,
    ledger_id: str,
    prior_attempt_count: int = 1,
) -> ledger_api.DatabricksClusterHourLedger:
    """Create paid predecessor accounting without retained campaign data.

    These deliberately unsubmitted attempts exercise the public reservation and
    terminal serializers. Qualification tests subsequently create their own
    receipt-bound jobs through the real submission controller and fake transport.
    """
    ledger = ledger_api.create_databricks_cluster_hour_ledger_json(
        path, ledger_id=ledger_id, cap_cluster_hours=1024.0
    )
    for index in range(prior_attempt_count):
        attempt_id = f"synthetic-predecessor-{index:03d}"
        payload = {
            "run_name": attempt_id,
            "timeout_seconds": 3600,
            "tasks": [
                {
                    "task_key": "synthetic-predecessor",
                    "timeout_seconds": 3600,
                    "max_retries": 0,
                    "new_cluster": {"num_workers": 0, "node_type_id": "g6.8xlarge"},
                    "spark_python_task": {
                        "python_file": "dbfs:/unit-fixtures/predecessor.py",
                    },
                }
            ],
        }
        ledger_api.reserve_databricks_run_attempt_json(
            path, payload, attempt_id=attempt_id, workload_id=attempt_id
        )
        ledger = ledger_api.record_databricks_run_terminal_actual_json(
            path,
            attempt_id=attempt_id,
            terminal_state="succeeded",
            actual_cluster_duration_seconds=60.0,
        )
    return ledger
