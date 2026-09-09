"""Synthetic logged failures matching the historical controller grammars.

These are unit-test inputs, never retained cloud responses or publication data.
The production validators still check all grammar, topology, and digest fields.
"""

import hashlib

import document_kv_cache.gpu_qualification_databricks as api


def configure_failure(family, job_id, plan_sha256, run, output):
    prefix = f"GPU_QUALIFICATION_{family}_FAILURE_"

    def pin(name):
        return getattr(api, prefix + name)

    work_root = f"{api.GPU_QUALIFICATION_LOCAL_WORK_ROOT}/{plan_sha256}/{job_id}"
    if family in {"BOOTSTRAP_FILE_GLOBAL", "BOOTSTRAP_CLUSTER_IDENTITY"}:
        error = pin("ERROR")
        output.update(
            error=error, error_trace=error + "\n", logs="synthetic driver log\n"
        )
        if family == "BOOTSTRAP_FILE_GLOBAL":
            output.pop("logs")
            output.pop("logs_truncated")
        return
    if family == "RUNTIME_LOCK_INDEX":
        lock = (
            "/local_disk0/.ephemeral_nfs/envs/pythonEnv-"
            "00000000-0000-4000-8000-000000000001/lib/python3.11/site-packages/"
            "document_kv_cache/runtime_locks/vllm-0.27.1-cu129-py311-manylinux_2_35.lock"
        )
        error = api._RUNTIME_LOCK_INDEX_FAILURE_NORMALIZED_ERROR.replace(
            "{runtime_python}", work_root + "/runtime/bin/python"
        ).replace("{runtime_lock}", lock)
        output.update(
            error=error, error_trace=error + "\n", logs=pin("LOG_MARKER") + "\n"
        )
        return
    if family == "SITE_PACKAGES_PATH":
        error = api._SITE_PACKAGES_PATH_FAILURE_NORMALIZED_ERROR.replace(
            "{invalid_site_packages}",
            work_root + "/runtime/local/lib/python3.11/dist-packages",
        )
        output.update(
            error=error,
            error_trace=pin("FREEZER_TRACE_MARKER") + "\n" + error,
            logs=(pin("PIP_CHECK_LOG_MARKER") + "\n") * 2,
        )
        return

    output["logs"] = (
        (pin("PIP_CHECK_LOG_MARKER") + "\n") * 2
        + pin("VIRTUALENV_LOG_PREFIX")
        + "123ms\n"
        + pin("ENSUREPIP_LOG_ARGV")
        + "\n"
    )
    if family == "RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS":
        observer = job_id in pin("PACKED_PAGE_ROUNDTRIP_JOB_IDS")
        template = (
            api._RUNTIME_OBSERVATION_FAILURE_NORMALIZED_ERROR
            if observer
            else api._WORKER_SUBPROCESS_FAILURE_NORMALIZED_ERROR
        )
        error = template.replace("{work_root}", work_root)
        markers = pin("OBSERVER_TRACE_MARKERS" if observer else "WORKER_TRACE_MARKERS")
        trace = "\n".join(markers) + "\n" + error
        if observer:
            trace += "\n" + pin("OBSERVER_TRACE_MARKER")
        else:
            trace += (
                "\n"
                + pin("WORKER_MODULE_MARKER")
                + "\nCalledProcessError\nCalledProcessError"
            )
        output.update(error=error, error_trace=trace)
        for record, state in [
            (run, "INTERNAL_ERROR"),
            (run["tasks"][0], "TERMINATED"),
            (output["metadata"], "TERMINATED"),
            (output["metadata"]["tasks"][0], "TERMINATED"),
        ]:
            record["state"] = {"life_cycle_state": state, "result_state": "FAILED"}
        return

    assert family == "MIXED_SENTINEL_AND_RESULT_VALIDATION"
    category = api._mixed_sentinel_and_result_validation_failure_categories()[job_id]
    output["logs"] += (work_root + "\n") * 12
    if category == "version_mismatch":
        error = f"ValueError: job result {job_id} vLLM version mismatch"
        trace = (
            "\n".join(
                marker
                for marker, count in pin("VERSION_TRACE_MARKERS")
                for _ in range(count)
            )
            + "\n"
            + error
        )
    else:
        marker_key = {
            "unresolved_native": "UNRESOLVED_NATIVE_ERROR_MARKERS",
            "layout_conflict": "LAYOUT_ERROR_MARKERS",
            "flashinfer": "FLASHINFER_ERROR_MARKERS",
        }[category]
        stderr = "\n".join(pin(marker_key)) + "\n" + work_root
        sha = hashlib.sha256(stderr.encode()).hexdigest()
        flash = category == "flashinfer"
        normal = (
            (
                f"RuntimeError: GPU sentinel '{job_id}' worker exited with status 1; "
                f"stdout(bytes={1 if flash else 0},sha256={sha if flash else hashlib.sha256(b'').hexdigest()},"
                f"truncated={'true' if flash else 'false'},tail={'repr' if flash else 'empty'}); "
                f"stderr(bytes={len(stderr.encode())},sha256={sha},"
                f"truncated={'false' if category == 'unresolved_native' else 'true'},tail='{stderr}')"
            )
            .replace("tail=repr", "tail='synthetic output'")
            .replace("tail=empty", "tail=''")
        )
        error = normal.replace(
            "RuntimeError", api._DATABRICKS_ANSI_RUNTIME_ERROR_HTML_PREFIX, 1
        )
        trace = (
            "\n".join(
                marker
                for marker, count in pin("WORKER_TRACE_MARKERS")
                for _ in range(count)
            )
            + "\n"
            + pin("WORKER_MODULE_MARKER")
            + "\n"
            + normal
        )
    output.update(error=error, error_trace=trace)
    task = run["tasks"][0]
    parent_message = f"Task {task['task_key']} failed with message: Workload failed, see run output for details."
    for record, state, message in [
        (run, "INTERNAL_ERROR", parent_message),
        (task, "TERMINATED", "Workload failed, see run output for details"),
        (
            output["metadata"],
            "TERMINATED",
            "Workload failed, see run output for details",
        ),
        (
            output["metadata"]["tasks"][0],
            "TERMINATED",
            "Workload failed, see run output for details",
        ),
    ]:
        record["state"] = {
            "life_cycle_state": state,
            "result_state": "FAILED",
            "state_message": message,
            "user_cancelled_or_timedout": False,
        }
        record["status"] = {
            "state": "TERMINATED",
            "termination_details": {
                "code": "RUN_EXECUTION_ERROR",
                "message": message,
                "type": "CLIENT_ERROR",
            },
        }
    output["metadata"]["tasks"][0]["attempt_number"] = 0
