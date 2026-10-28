import json
import csv
from pathlib import Path
import pytest

from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec
from simple_adp.runner.state import JobRuntime, JobStateStore, RunStatus
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.scheduler import ControllerMemory, InvocationCounters, SuiteCoordinator
from types import SimpleNamespace


def _runtime(tmp_path, label, *, status, exit_code=None, ended_at=0.0, model="ADP-W-All", profile=None):
    identity = RunIdentity(
        provider="toy", suite="hard", task_id="task/1", split_repeat=0,
        fold=0, sample=0, suite_repeat=2, model_route=label, seed=42,
        split_sha256="split", run_signature="suite-signature",
    )
    route = ModelRoute(model, generic_profile=profile, init_width=8, init_depth=2)
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "hard", "task/1"), "simple_adp", route,
        "split.npz", str(tmp_path / "results" / label), {}, ("python", "worker.py"),
    )
    return JobRuntime(
        spec, status=status, exit_code=exit_code,
        last_exit_code=exit_code, ended_at=ended_at,
    )


def test_controller_memory_default_record_has_exact_ledger_fields_and_values():
    memory = ControllerMemory.defaults(RunnerSettings()).values

    assert memory == {
        "batch_scale": 1.0,
        "pressure_backoff_pending": False,
        "last_backoff_reason": None,
        "last_backoff_ts": None,
        "cpu_launches_enabled": True,
        "gpu_launches_enabled": True,
        "host_gate_closed": False,
        "gpu_gate_closed": False,
        "host_gate_lock_job_id": None,
        "gpu_gate_lock_job_id": None,
        "host_gate_buffer_mib": 500.0,
        "gpu_gate_buffer_mib": 100.0,
        "host_hold_until": 0.0,
        "gpu_hold_until": 0.0,
        "cpu_peak_host_mib": 0.0,
        "gpu_peak_vram_mib": 0.0,
        "last_child_host_mib": 0.0,
        "last_child_gpu_mib": 0.0,
        "last_host_remaining_mib": 0.0,
        "last_gpu_remaining_mib": 0.0,
        "last_child_job_id": None,
        "external_host_mib": 0.0,
        "external_gpu_mib": 0.0,
        "external_host_peak_mib": 0.0,
        "external_gpu_peak_mib": 0.0,
        "external_host_pids": [],
        "external_gpu_pids": [],
    }


def test_job_runtime_record_serializes_every_durable_field_but_not_live_handles(tmp_path, monkeypatch):
    import re
    import simple_adp.runner.state as state_module

    monkeypatch.setattr(state_module, "local_timestamp", lambda: "20261021_150405")
    runtime = _runtime(tmp_path, "runtime-record", status=RunStatus.RUNNING, model="ADP-WD-RR-D1", profile="faithful")
    runtime.attempt = 3
    runtime.retry_count = 2
    runtime.pid = 4321
    runtime.device_mode = "gpu"
    runtime.pause_reason = "gpu_pressure"
    runtime.pause_count = 4
    runtime.gpu_pause_count = 3
    runtime.host_pause_count = 1
    runtime.force_cpu_after_cuda_oom = True
    runtime.exit_code = 7
    runtime.last_exit_code = 9
    runtime.started_at = 10.0
    runtime.ended_at = 11.0
    runtime.last_progress_ts = 12.0
    runtime.last_error = "last attempt failed"
    runtime.launch_context = {"device": "gpu", "attempt": 3}
    runtime.recovered_from_status = "paused"
    runtime.recovered_at = "20261021_150400"
    runtime.log_path = "/tmp/runtime.log"
    runtime.resource_snapshot = {"gpu_free_mib": 123.0}
    runtime.unknown_state_fields = {"legacy_field": "retained"}
    runtime.process = object()
    runtime.log_handle = object()

    record = runtime.to_record()

    assert set(record) == {
        "legacy_field", "job_id", "identity", "route", "task_ref", "command",
        "result_dir", "split_file", "attempt", "retry_count", "status", "pid",
        "device_mode", "pause_reason", "pause_count", "gpu_pause_count",
        "host_pause_count", "force_cpu_after_cuda_oom", "exit_code", "last_exit_code",
        "started_at", "ended_at", "last_progress_ts", "last_error", "launch_context",
        "resource_snapshot", "log_path", "updated_at", "recovered_from_status", "recovered_at",
    }
    assert record["job_id"] == runtime.job_id
    assert record["identity"] == runtime.spec.identity.as_dict()
    assert record["route"] == runtime.spec.route.as_dict()
    assert record["task_ref"] == {
        "provider": "toy", "suite": "hard", "task_id": "task/1",
        "dataset_id": None, "metadata": None,
    }
    assert record["command"] == ["python", "worker.py"]
    assert record["status"] == "running"
    assert record["attempt"] == 3 and record["retry_count"] == 2
    assert record["last_exit_code"] == 9 and record["exit_code"] == 7
    assert record["legacy_field"] == "retained"
    assert record["recovered_from_status"] == "paused"
    assert record["recovered_at"] == "20261021_150400"
    assert record["updated_at"] == "20261021_150405"
    assert not re.search(r"process|log_handle", " ".join(record))


def test_compatibility_views_aggregate_sorted_states_and_source_status_rules(tmp_path, monkeypatch):
    import simple_adp.runner.state as state_module

    source_timestamp = "20261021_131900"
    monkeypatch.setattr(state_module, "local_timestamp", lambda: source_timestamp)
    store = JobStateStore(tmp_path / "run")
    candidates = [
        _runtime(tmp_path, "candidate-a", status=RunStatus.COMPLETED, exit_code=0, ended_at=10),
        _runtime(tmp_path, "candidate-b", status=RunStatus.COMPLETED, exit_code=1, ended_at=99),
        _runtime(tmp_path, "candidate-c", status=RunStatus.LAUNCHING),
        _runtime(tmp_path, "candidate-d", status=RunStatus.RETRYING),
        _runtime(tmp_path, "candidate-f", status=RunStatus.COMPLETED, exit_code=0, ended_at=20),
        _runtime(tmp_path, "candidate-e", status=RunStatus.COMPLETED, exit_code=0, ended_at=20, profile="corrected"),
    ]
    for candidate in candidates:
        store.save(candidate)

    first = candidates[0]
    first.gpu_pause_count = 2
    first.host_pause_count = 3
    first.force_cpu_after_cuda_oom = True
    store.save(first)
    identity = first.spec.identity
    group = store._compatibility_group(identity.as_dict(), first.spec.route)
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    task = json.loads((group["task_root"] / "task_state.json").read_text())
    replay = json.loads((group["task_root"] / "replay_manifest.json").read_text())
    candidate_state = json.loads((store.job_dir(first) / "candidate_state.json").read_text())
    candidate_summary = json.loads((store.job_dir(first) / "candidate_summary.json").read_text())

    assert set(candidate_state) == {
        "schema", "scope", "job_id", "identity", "route", "task_ref",
        "candidate_dir", "results_dir", "state_path", "architecture", "status",
        "completed", "attempt", "retry_count", "device_mode", "pause_count",
        "gpu_pause_count", "host_pause_count", "force_cpu_after_cuda_oom",
        "pause_reason", "command", "exit_code", "last_exit_code", "started_at",
        "ended_at", "last_error", "launch_context", "resource_snapshot", "updated_at",
    }
    assert candidate_state["scope"] == "candidate"
    assert candidate_state["schema"] == 1
    assert candidate_state["identity"] == identity.as_dict()
    assert candidate_state["route"] == first.spec.route.as_dict()
    assert candidate_state["task_ref"] == {
        "provider": "toy", "suite": "hard", "task_id": "task/1",
        "dataset_id": None, "metadata": None,
    }
    assert candidate_state["architecture"] == {"depth": 2, "width": 8}
    assert candidate_state["completed"] is True
    assert candidate_state["attempt"] == 0 and candidate_state["retry_count"] == 0
    assert candidate_state["device_mode"] is None
    assert candidate_state["gpu_pause_count"] == 2
    assert candidate_state["host_pause_count"] == 3
    assert candidate_state["force_cpu_after_cuda_oom"] is False
    assert store.load(first.job_id)["force_cpu_after_cuda_oom"] is True
    assert store.load(first.job_id)["updated_at"] == source_timestamp
    assert candidate_state["updated_at"] == source_timestamp
    assert set(candidate_summary) == {
        "schema", "job_id", "completed", "status", "exit_code", "reason",
        "attempt", "device_mode", "ended_at", "results_dir", "state_path",
        "updated_at",
    }
    assert candidate_summary == {
        "schema": 1, "job_id": first.job_id, "completed": True,
        "status": "completed", "exit_code": 0, "reason": None,
        "attempt": 0, "device_mode": None,
        "ended_at": 10, "results_dir": first.spec.result_dir,
        "state_path": str(store.state_path(first)),
        "updated_at": source_timestamp,
    }
    assert method["updated_at"] == task["updated_at"] == replay["updated_at"] == source_timestamp
    assert set(method) == {
        "schema", "scope", "provider", "suite", "task_id", "repeat", "model_id",
        "generic_profile", "candidate_count", "counts", "latest_completed",
        "candidate_root", "candidate_states", "best_candidate_dir", "updated_at",
    }
    assert set(task) == {
        "schema", "scope", "provider", "suite", "task_id", "repeat", "family_count",
        "candidate_count", "counts", "latest_completed", "task_root", "method_roots",
        "best_candidate_dir", "updated_at",
    }
    assert set(replay) == {
        "schema", "provider", "suite", "task_id", "repeat", "seed", "task_root",
        "method_root", "candidate_dir", "candidates", "updated_at",
    }
    assert method["candidate_count"] == 5
    assert method["counts"] == {"completed": 3, "failed": 0, "paused": 0, "queued": 1, "running": 1}
    assert method["latest_completed"]["job_id"] == candidates[4].job_id  # ended=20 is latest in this profile
    assert method["best_candidate_dir"] == candidates[0].spec.result_dir  # sorted first successful, not metric-ranked
    assert method["schema"] == task["schema"] == replay["schema"] == 1
    assert method["scope"] == "method" and task["scope"] == "task"
    assert method["provider"] == task["provider"] == replay["provider"] == "toy"
    assert method["suite"] == task["suite"] == replay["suite"] == "hard"
    assert method["task_id"] == task["task_id"] == replay["task_id"] == "task%2F1"
    assert method["repeat"] == task["repeat"] == replay["repeat"] == 2
    assert method["model_id"] == "ADP-W-All" and method["generic_profile"] is None
    assert method["candidate_root"] == str(group["method_root"])
    assert method["candidate_states"] == sorted(
        str(store.state_path(item)) for item in (candidates[0], candidates[1], candidates[2], candidates[3], candidates[4])
    )
    expected_latest = store.load(candidates[4].job_id)
    expected_latest["state_path"] = str(store.state_path(candidates[4]))
    assert method["latest_completed"] == expected_latest
    assert {key: value for key, value in method.items() if key != "updated_at"} == {
        "schema": 1, "scope": "method", "provider": "toy", "suite": "hard",
        "task_id": "task%2F1", "repeat": 2, "model_id": "ADP-W-All",
        "generic_profile": None, "candidate_count": 5,
        "counts": {"completed": 3, "failed": 0, "paused": 0, "queued": 1, "running": 1},
        "latest_completed": expected_latest,
        "candidate_root": str(group["method_root"]),
        "candidate_states": sorted(
            str(store.state_path(item)) for item in candidates[:5]
        ),
        "best_candidate_dir": candidates[0].spec.result_dir,
    }
    assert task["candidate_count"] == 6
    assert task["family_count"] == 2  # faithful/default and corrected are separate method families
    assert task["counts"] == {"completed": 4, "failed": 0, "paused": 0, "queued": 1, "running": 1}
    assert task["latest_completed"]["job_id"] == candidates[4].job_id
    assert task["task_root"] == str(group["task_root"])
    assert task["method_roots"] == sorted(str(store._compatibility_group(
        item.spec.identity.as_dict(), item.spec.route,
    )["method_root"]) for item in (candidates[0], candidates[5]))
    assert task["best_candidate_dir"] == candidates[0].spec.result_dir
    assert {key: value for key, value in task.items() if key != "updated_at"} == {
        "schema": 1, "scope": "task", "provider": "toy", "suite": "hard",
        "task_id": "task%2F1", "repeat": 2, "family_count": 2, "candidate_count": 6,
        "counts": {"completed": 4, "failed": 0, "paused": 0, "queued": 1, "running": 1},
        "latest_completed": expected_latest,
        "task_root": str(group["task_root"]),
        "method_roots": task["method_roots"],
        "best_candidate_dir": candidates[0].spec.result_dir,
    }
    assert len(replay["candidates"]) == 6
    assert {entry["job_id"] for entry in replay["candidates"]} == {item.job_id for item in candidates}
    assert replay["seed"] == identity.seed
    assert replay["task_root"] == str(group["task_root"])
    assert replay["method_root"] == str(group["method_root"])
    assert replay["candidate_dir"] == str(store.job_dir(first))
    assert all(set(entry) == {
        "job_id", "model_id", "generic_profile", "route_label", "seed", "candidate_dir",
        "state_path", "status", "attempt", "device_mode", "exit_code",
    } for entry in replay["candidates"])
    expected_replay_candidates = []
    for candidate in sorted(candidates, key=lambda item: str(store.state_path(item))):
        state = store.load(candidate.job_id)
        route_payload = candidate.spec.route.as_dict()
        expected_replay_candidates.append({
            "job_id": state["job_id"],
            "model_id": route_payload["model_id"],
            "generic_profile": route_payload["generic_profile"],
            "route_label": route_payload["route_label"],
            "seed": state["identity"]["seed"],
            "candidate_dir": state["result_dir"],
            "state_path": str(store.state_path(candidate)),
            "status": state["status"],
            "attempt": state["attempt"],
            "device_mode": state["device_mode"],
            "exit_code": state["last_exit_code"],
        })
    assert replay["candidates"] == expected_replay_candidates


def test_compatibility_views_retain_source_numeric_like_string_sorting(tmp_path):
    store = JobStateStore(tmp_path / "run")
    older = _runtime(tmp_path, "old", status=RunStatus.COMPLETED, exit_code=0, ended_at=2)
    newer = _runtime(tmp_path, "new", status=RunStatus.COMPLETED, exit_code=0, ended_at=12)
    store.save(older)
    store.save(newer)
    group = store._compatibility_group(older.spec.identity.as_dict(), older.spec.route)
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    # The source recognizes numeric-like timestamps, then sorts their original
    # strings; lexically "2" sorts after "12".
    assert method["latest_completed"]["job_id"] == older.job_id


def test_compatibility_views_keep_the_source_permissive_numeric_predicate(tmp_path):
    store = JobStateStore(tmp_path / "run")
    numeric_like = _runtime(
        tmp_path, "dot-leading", status=RunStatus.COMPLETED,
        exit_code=0, ended_at=".5",
    )
    nonnumeric = _runtime(
        tmp_path, "letters", status=RunStatus.COMPLETED,
        exit_code=0, ended_at="z",
    )
    store.save(numeric_like)
    store.save(nonnumeric)
    group = store._compatibility_group(
        numeric_like.spec.identity.as_dict(), numeric_like.spec.route,
    )
    method = json.loads((group["method_root"] / "method_state.json").read_text())

    # The reference removes one dot before isdigit(), classifying .5 as numeric
    # and preferring it to the otherwise lexically greater nonnumeric string.
    assert method["latest_completed"]["job_id"] == numeric_like.job_id


def test_compatibility_views_use_job_id_tie_break_but_first_sorted_success_for_best_dir(
    tmp_path,
):
    store = JobStateStore(tmp_path / "run")
    candidates = [
        _runtime(
            tmp_path, label, status=RunStatus.COMPLETED, exit_code=0,
            ended_at="7",
        )
        for label in ("tie-alpha", "tie-omega")
    ]
    for candidate in candidates:
        store.save(candidate)

    group = store._compatibility_group(
        candidates[0].spec.identity.as_dict(), candidates[0].spec.route,
    )
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    by_state_path = sorted(candidates, key=lambda item: str(store.state_path(item)))

    # Equal numeric-like timestamps use job ID as the final latest-success
    # tie-break; best_candidate_dir remains the first successful state found.
    assert method["latest_completed"]["job_id"] == by_state_path[-1].job_id
    assert method["best_candidate_dir"] == by_state_path[0].spec.result_dir


def test_compatibility_grouping_uses_task_repeat_and_method_identity_only(tmp_path):
    from dataclasses import replace

    store = JobStateStore(tmp_path / "run")
    base = _runtime(
        tmp_path, "base", status=RunStatus.COMPLETED, exit_code=0, ended_at=4,
    )
    other_split = _runtime(
        tmp_path, "other-split", status=RunStatus.COMPLETED, exit_code=0, ended_at=5,
    )
    other_split.spec = replace(
        other_split.spec,
        identity=replace(
            other_split.spec.identity,
            split_repeat=7,
            fold=3,
            sample=2,
            model_route="different-route-label",
            seed=101,
        ),
    )
    other_profile = _runtime(
        tmp_path, "other-profile", status=RunStatus.FAILED, exit_code=2,
    )
    other_profile.spec = replace(
        other_profile.spec,
        identity=replace(other_profile.spec.identity, model_route="corrected-route"),
        route=replace(other_profile.spec.route, generic_profile="corrected"),
    )
    other_repeat = _runtime(
        tmp_path, "other-repeat", status=RunStatus.COMPLETED, exit_code=0, ended_at=9,
    )
    other_repeat.spec = replace(
        other_repeat.spec,
        identity=replace(other_repeat.spec.identity, suite_repeat=3),
    )
    other_task = _runtime(
        tmp_path, "other-task", status=RunStatus.COMPLETED, exit_code=0, ended_at=10,
    )
    other_task.spec = replace(
        other_task.spec,
        identity=replace(other_task.spec.identity, task_id="task/2"),
    )

    for runtime in (base, other_split, other_profile, other_repeat, other_task):
        store.save(runtime)

    # The source counts unknown state strings as queued. The task aggregate
    # groups across route profiles and split/seed coordinates, but not across
    # suite repeats or task IDs; method identity additionally separates the
    # generic profile.
    path = store.state_path(other_profile)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["status"] = "future-status"
    path.write_text(json.dumps(record), encoding="utf-8")
    store._write_compatibility_views(base)

    group = store._compatibility_group(base.spec.identity.as_dict(), base.spec.route)
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    task = json.loads((group["task_root"] / "task_state.json").read_text())
    replay = json.loads((group["task_root"] / "replay_manifest.json").read_text())

    assert method["candidate_count"] == 2
    assert method["counts"] == {
        "completed": 2, "running": 0, "paused": 0, "failed": 0, "queued": 0,
    }
    assert method["latest_completed"]["job_id"] == other_split.job_id
    assert task["candidate_count"] == 3
    assert task["family_count"] == 2
    assert task["counts"] == {
        "completed": 2, "running": 0, "paused": 0, "failed": 0, "queued": 1,
    }
    assert {entry["job_id"] for entry in replay["candidates"]} == {
        base.job_id, other_split.job_id, other_profile.job_id,
    }


def test_compatibility_views_leave_success_fields_empty_without_zero_exit(tmp_path):
    store = JobStateStore(tmp_path / "run")
    failed = _runtime(
        tmp_path, "failed-only", status=RunStatus.FAILED, exit_code=2, ended_at=12,
    )
    store.save(failed)
    group = store._compatibility_group(failed.spec.identity.as_dict(), failed.spec.route)
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    task = json.loads((group["task_root"] / "task_state.json").read_text())
    replay = json.loads((group["task_root"] / "replay_manifest.json").read_text())
    candidate_summary = json.loads((store.job_dir(failed) / "candidate_summary.json").read_text())

    assert method["candidate_count"] == task["candidate_count"] == 1
    assert method["counts"] == task["counts"] == {
        "completed": 0, "running": 0, "paused": 0, "failed": 1, "queued": 0,
    }
    assert method["latest_completed"] is None
    assert task["latest_completed"] is None
    assert method["best_candidate_dir"] is None
    assert task["best_candidate_dir"] is None
    assert candidate_summary["completed"] is False
    assert candidate_summary["status"] == "failed"
    assert len(replay["candidates"]) == 1
    assert replay["candidates"][0]["exit_code"] == 2


def test_compatibility_aggregation_includes_persisted_state_outside_manifest(tmp_path):
    store = JobStateStore(tmp_path / "run")
    unplanned = _runtime(
        tmp_path, "persisted-but-unplanned", status=RunStatus.COMPLETED,
        exit_code=0, ended_at=7,
    )
    unplanned_state_path = store.state_path(unplanned)
    unplanned_state_path.parent.mkdir(parents=True, exist_ok=True)
    unplanned_record = unplanned.to_record()
    unplanned_record["state_path"] = str(unplanned_state_path)
    unplanned_state_path.write_text(json.dumps(unplanned_record), encoding="utf-8")

    planned = _runtime(
        tmp_path, "manifest-trigger", status=RunStatus.QUEUED,
    )
    # Saving a planned job regenerates grouped views by scanning persisted
    # job-state files, including resumable/orphaned states outside this plan.
    store.save(planned)

    group = store._compatibility_group(planned.spec.identity.as_dict(), planned.spec.route)
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    task = json.loads((group["task_root"] / "task_state.json").read_text())
    replay = json.loads((group["task_root"] / "replay_manifest.json").read_text())

    assert method["candidate_count"] == task["candidate_count"] == 2
    assert method["counts"]["completed"] == task["counts"]["completed"] == 1
    assert method["latest_completed"]["job_id"] == unplanned.job_id
    assert {entry["job_id"] for entry in replay["candidates"]} == {
        planned.job_id, unplanned.job_id,
    }


def test_compatibility_success_and_replay_use_last_exit_code(tmp_path):
    store = JobStateStore(tmp_path / "run")
    completed = _runtime(
        tmp_path, "source-success-field", status=RunStatus.COMPLETED,
        exit_code=7, ended_at=12,
    )
    completed.last_exit_code = 0
    store.save(completed)
    group = store._compatibility_group(
        completed.spec.identity.as_dict(), completed.spec.route,
    )
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    replay = json.loads((group["task_root"] / "replay_manifest.json").read_text())
    candidate_state = json.loads(
        (store.job_dir(completed) / "candidate_state.json").read_text()
    )
    assert candidate_state["completed"] is True
    assert method["latest_completed"]["job_id"] == completed.job_id
    assert replay["candidates"][0]["exit_code"] == 0

    contradictory = _runtime(
        tmp_path, "source-failure-field", status=RunStatus.COMPLETED,
        exit_code=0, ended_at=13,
    )
    contradictory.last_exit_code = 2
    store.save(contradictory)
    assert store.load(contradictory.job_id)["last_exit_code"] == 2
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    candidate_state = json.loads(
        (store.job_dir(contradictory) / "candidate_state.json").read_text()
    )
    assert candidate_state["completed"] is False
    assert method["latest_completed"]["job_id"] == completed.job_id


def test_compatibility_status_and_exit_code_truth_table_matches_ledger(tmp_path):
    """Pin status counts and success selection independently across corrupt values."""
    store = JobStateStore(tmp_path / "run")
    trigger = _runtime(tmp_path, "truth-table-trigger", status=RunStatus.QUEUED)
    statuses = (None, "", "launching", "completed", "failed", "paused", "retrying", "future")
    exit_codes = (None, "0", "bad", 0, 1, 0.5, False, True)
    expected_counts = {key: 0 for key in ("completed", "running", "paused", "failed", "queued")}
    successful: list[tuple[str, str]] = []
    fixture_paths = []
    index = 0
    for status in statuses:
        normalized_status = str(status or "queued")
        count_key = "running" if normalized_status == "launching" else normalized_status
        if count_key not in expected_counts:
            count_key = "queued"
        for exit_code in exit_codes:
            expected_counts[count_key] += 1
            job_id = f"truth-{index:03d}"
            path = store.jobs_root / f"fixture-{index:03d}" / "job_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = trigger.to_record()
            record.update({
                "job_id": job_id,
                "identity": trigger.spec.identity.as_dict(),
                "route": trigger.spec.route.as_dict(),
                "status": status,
                "exit_code": 99,  # Aggregation must use last_exit_code.
                "last_exit_code": exit_code,
                "ended_at": "8",  # Exercise job-ID tie-breaking among successes.
                "result_dir": f"results/{job_id}",
                "state_path": str(path),
            })
            path.write_text(json.dumps(record), encoding="utf-8")
            fixture_paths.append((str(path), job_id, record))
            try:
                parsed_exit = int(exit_code)
            except Exception:
                parsed_exit = 1
            if status == "completed" and parsed_exit == 0:
                successful.append((str(path), job_id))
            index += 1

    store._write_compatibility_views(trigger)
    group = store._compatibility_group(trigger.spec.identity.as_dict(), trigger.spec.route)
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    replay = json.loads((group["task_root"] / "replay_manifest.json").read_text())

    assert method["candidate_count"] == len(statuses) * len(exit_codes)
    assert method["counts"] == expected_counts
    expected_latest = max(successful, key=lambda item: item[1])
    assert method["latest_completed"]["job_id"] == expected_latest[1]
    first_success_by_state_path = min(successful, key=lambda item: item[0])
    assert method["best_candidate_dir"] == f"results/{first_success_by_state_path[1]}"
    assert [row["job_id"] for row in replay["candidates"]] == [
        job_id for _path, job_id, _record in sorted(fixture_paths)
    ]


def test_final_reports_and_progress_totals_match_runner_record_contract(tmp_path):
    runtime = _runtime(tmp_path, "report-route", status=RunStatus.COMPLETED, exit_code=0, ended_at=2)
    runtime.last_exit_code = 0
    plan = SimpleNamespace(run_signature="suite-signature", runs=(runtime.spec,))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(run_hardness_screen=False), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    coordinator.state_store.save(runtime)
    coordinator.counters = InvocationCounters(completed=1, failed_attempts=2, retries=1)
    telemetry = tmp_path / "results" / "report-route" / "telemetry" / "progress_state.json"
    telemetry.parent.mkdir(parents=True)
    telemetry.write_text(json.dumps({"history_rows": 1, "final_epoch": 1}))

    summary = coordinator._write_final_reports()
    report = json.loads((tmp_path / "run" / "final_report.json").read_text())
    suite = json.loads((tmp_path / "run" / "suite_summary.json").read_text())
    assert report == suite == summary
    assert set(summary) == {
        "run_root", "results_dir", "run_signature", "manifest_jobs", "completed",
        "terminal_failures", "failed_attempts", "retries", "recovered_unknown_exit",
        "timestamp", "resource_snapshot", "hardness_screening", "batches_done",
        "epochs_done", "live_batches_done", "live_epochs_done", "telemetry_jobs",
        "telemetry_dirs",
    }
    assert (summary["batches_done"], summary["epochs_done"]) == (1, 1)
    assert (summary["live_batches_done"], summary["live_epochs_done"]) == (0, 0)
    final_markdown = (tmp_path / "run" / "final_report.md").read_text()
    suite_markdown = (tmp_path / "run" / "suite_summary.md").read_text()
    assert "Completed: 1" in final_markdown
    assert f"Total jobs: 1" in final_markdown
    assert f"Failed attempts: 2" in final_markdown
    assert "Batches done: 1" in final_markdown and "Epochs done: 1" in final_markdown
    assert "Live-job batches done: 0" in final_markdown and "Live-job epochs done: 0" in final_markdown
    assert f"Timestamp: {summary['timestamp']}" in final_markdown
    assert "Batches done: 1" in suite_markdown and "Epochs done: 1" in suite_markdown
    assert "Timestamp:" not in suite_markdown
    with (tmp_path / "run" / "task_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert tuple(rows[0]) == (
        "job_id", "status", "attempt", "retry_count", "exit_code",
        "last_exit_code", "result_dir",
    )
    assert rows[0] == {
        "job_id": runtime.job_id,
        "status": "completed",
        "attempt": "0",
        "retry_count": "0",
        "exit_code": "0",
        "last_exit_code": "0",
        "result_dir": runtime.spec.result_dir,
    }


@pytest.mark.parametrize(("success", "exit_code", "progress_event"), [
    (True, 0, "complete"),
    (False, 2, "fail"),
])
def test_finalization_separates_terminal_progress_rows_from_lifecycle_events(
    tmp_path, success, exit_code, progress_event,
):
    runtime = _runtime(
        tmp_path, "final-progress", status=RunStatus.RUNNING, exit_code=exit_code,
    )
    runtime.attempt = 3
    runtime.device_mode = "gpu"
    plan = SimpleNamespace(run_signature="suite-signature", runs=(runtime.spec,))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(run_hardness_screen=False), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    snapshot = SimpleNamespace(as_dict=lambda: {"host_available_mib": 1234})

    coordinator._finalize(runtime, success=success, reason=None if success else "nonzero_exit:2", snapshot=snapshot)

    with coordinator.state_store.progress_path.open(newline="", encoding="utf-8") as stream:
        progress = list(csv.DictReader(stream))
    assert len(progress) == 1
    row = progress[0]
    assert tuple(row) == JobStateStore.PROGRESS_FIELDS
    assert row["event"] == progress_event
    assert row["job_id"] == runtime.job_id
    assert (row["provider"], row["suite"], row["task_id"]) == ("toy", "hard", "task/1")
    assert (row["split_repeat"], row["fold"], row["sample"], row["suite_repeat"], row["seed"]) == (
        "0", "0", "0", "2", "42",
    )
    assert row["model_id"] == "ADP-W-All"
    assert row["route_label"] == runtime.spec.route.route_label
    assert row["attempt"] == "3" and row["device_mode"] == "gpu"
    assert row["exit_code"] == str(exit_code)
    assert json.loads(row["architecture"]) == {"depth": 2, "width": 8}
    assert row["results_dir"] == runtime.spec.result_dir
    assert json.loads(row["resource_snapshot"]) == {"host_available_mib": 1234}

    with coordinator.state_store.events_path.open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    assert [event["event"] for event in events] == ["finalize"]


def test_both_report_paths_atomically_replace_markdown_files(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module

    original = scheduler_module.atomic_write_text
    writes = []

    def record_atomic_write(path, content):
        writes.append(path.name)
        original(path, content)

    monkeypatch.setattr(scheduler_module, "atomic_write_text", record_atomic_write)
    runtime = _runtime(tmp_path, "atomic-report-route", status=RunStatus.COMPLETED, exit_code=0, ended_at=2)
    plan = SimpleNamespace(run_signature="suite-signature", runs=(runtime.spec,))
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(run_hardness_screen=False),
        run_root=tmp_path / "run",
        results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )

    coordinator._write_final_reports()
    persisted = coordinator.write_persisted_state_report()

    assert writes == [
        "final_report.md", "suite_summary.md",
        "final_report.md", "suite_summary.md",
    ]
    for filename in ("final_report.md", "suite_summary.md"):
        assert (tmp_path / "run" / filename).is_file()
        assert not (tmp_path / "run" / f"{filename}.tmp").exists()
    final_markdown = (tmp_path / "run" / "final_report.md").read_text()
    suite_markdown = (tmp_path / "run" / "suite_summary.md").read_text()
    assert "Batches done:" in final_markdown and "Epochs done:" in final_markdown
    assert f"Timestamp: {persisted['timestamp']}" in final_markdown
    assert "Batches done:" in suite_markdown and "Epochs done:" in suite_markdown
    assert "Timestamp:" not in suite_markdown


def test_controller_invocation_resets_transient_backoff_and_gate_state(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    state_path = run_root / "batch_backoff_state.json"
    state_path.write_text(json.dumps({
        "batch_scale": 0.125,
        "pressure_backoff_pending": True,
        "last_backoff_reason": "cuda_oom",
        "last_backoff_ts": 123.0,
        "cpu_launches_enabled": False,
        "gpu_launches_enabled": False,
        "host_gate_closed": True,
        "gpu_gate_closed": True,
        "host_gate_lock_job_id": "old-host",
        "gpu_gate_lock_job_id": "old-gpu",
        "host_gate_buffer_mib": 1,
        "gpu_gate_buffer_mib": 2,
        "host_hold_until": 999.0,
        "gpu_hold_until": 999.0,
        "cpu_peak_host_mib": 88,
        "gpu_peak_vram_mib": 77,
        "last_child_host_mib": 66,
        "last_child_gpu_mib": 55,
        "external_host_mib": 44,
        "external_gpu_mib": 33,
        "external_host_peak_mib": 22,
        "external_gpu_peak_mib": 11,
        "external_host_pids": [101],
        "external_gpu_pids": [202],
        "unrecognized_persisted_field": "preserved",
    }))
    settings = RunnerSettings(
        child_host_gate_buffer_mib=700, child_gpu_gate_buffer_mib=300,
    )
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="sig", runs=()), settings,
        run_root=run_root, results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )

    coordinator._load_controller_state()

    state = json.loads(state_path.read_text())
    assert coordinator.admission.batch_scale == 1.0
    assert coordinator.admission.pressure_backoff_pending is False
    assert coordinator.admission.last_backoff_reason is None
    assert coordinator.admission.cpu_launches_enabled is True
    assert coordinator.admission.gpu_launches_enabled is True
    assert coordinator.admission.host_gate_closed is False
    assert coordinator.admission.gpu_gate_closed is False
    assert state["host_gate_lock_job_id"] is None
    assert state["gpu_gate_lock_job_id"] is None
    assert state["host_gate_buffer_mib"] == 700
    assert state["gpu_gate_buffer_mib"] == 300
    assert state["host_hold_until"] == state["gpu_hold_until"] == 0.0
    assert state["cpu_peak_host_mib"] == state["gpu_peak_vram_mib"] == 0.0
    assert state["last_child_host_mib"] == state["last_child_gpu_mib"] == 0.0
    assert state["external_host_mib"] == state["external_gpu_mib"] == 0.0
    assert state["external_host_peak_mib"] == state["external_gpu_peak_mib"] == 0.0
    assert state["external_host_pids"] == state["external_gpu_pids"] == []
    assert state["unrecognized_persisted_field"] == "preserved"


def test_event_csv_appends_exact_schema_with_local_timestamp_and_best_effort_fsync(tmp_path, monkeypatch):
    import re
    import simple_adp.runner.state as state_module

    runtime = _runtime(tmp_path, "event-route", status=RunStatus.RUNNING)
    runtime.attempt = 3
    runtime.retry_count = 2
    runtime.device_mode = "gpu"
    runtime.pid = 4321
    store = JobStateStore(tmp_path / "run")
    fsync_calls = []

    def fail_fsync(fd):
        fsync_calls.append(fd)
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(state_module.os, "fsync", fail_fsync)
    store.append_event(runtime, "pause", reason=None, snapshot={"gpu_free_mib": 12.5})
    store.append_event(runtime, "resume", reason="external_drop", snapshot={"gpu_free_mib": 50})

    with store.events_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    raw_lines = store.events_path.read_text(encoding="utf-8").splitlines()
    assert raw_lines.count(",".join(JobStateStore.EVENT_FIELDS)) == 1
    assert len(rows) == 2
    assert tuple(rows[0]) == JobStateStore.EVENT_FIELDS
    assert re.fullmatch(r"\d{8}_\d{6}", rows[0]["timestamp"])
    assert (rows[0]["event"], rows[0]["reason"], rows[0]["pid"]) == ("pause", "", "4321")
    assert json.loads(rows[0]["resource_snapshot"]) == {"gpu_free_mib": 12.5}
    assert (rows[1]["event"], rows[1]["reason"]) == ("resume", "external_drop")
    assert json.loads(rows[1]["resource_snapshot"]) == {"gpu_free_mib": 50}
    assert len(fsync_calls) == 2


def test_event_csv_existing_empty_file_keeps_missing_header_and_falsey_snapshot_falls_back(tmp_path):
    runtime = _runtime(tmp_path, "empty-event-log", status=RunStatus.RUNNING)
    runtime.resource_snapshot = {"fallback": True}
    store = JobStateStore(tmp_path / "run")
    store.events_path.parent.mkdir(parents=True, exist_ok=True)
    store.events_path.touch()

    store.append_event(runtime, "sample", snapshot={})

    with store.events_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1
    assert len(rows[0]) == len(JobStateStore.EVENT_FIELDS)
    assert rows[0][0] != "timestamp"  # header is emitted only if the path did not exist
    assert json.loads(rows[0][-1]) == {"fallback": True}  # {} uses the runtime snapshot


def test_invalid_utf8_state_files_follow_broad_json_loader_fallbacks(tmp_path):
    from simple_adp.runner.state import JobStateStore

    run_root = tmp_path / "run"
    store = JobStateStore(run_root)
    corrupt_path = store.jobs_root / "unreadable" / "job_state.json"
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"\xff\xfe")
    assert store.load("unreadable") is None

    # Compatibility aggregation scans nested state files and skips any unreadable
    # record, then still writes the new candidate's views.
    candidate = _runtime(tmp_path, "utf8-fallback", status=RunStatus.QUEUED)
    store.save(candidate)
    method_path = next((run_root / "tasks").rglob("method_state.json"))
    method = json.loads(method_path.read_text(encoding="utf-8"))
    assert method["candidate_count"] == 1

    state_path = run_root / "batch_backoff_state.json"
    state_path.write_bytes(b"\xff\xfe")
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="sig", runs=()), RunnerSettings(),
        run_root=run_root, results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    coordinator._load_controller_state()
    restored = json.loads(state_path.read_text(encoding="utf-8"))
    assert restored["batch_scale"] == 1.0
    assert restored["host_gate_closed"] is False


def test_compatibility_views_skip_malformed_nested_identity_and_route_shapes(tmp_path):
    store = JobStateStore(tmp_path / "run")
    current = _runtime(tmp_path, "well-formed", status=RunStatus.QUEUED)
    legacy_root = store.jobs_root / "legacy"
    legacy_root.mkdir(parents=True)
    (legacy_root / "job_state.json").write_text(json.dumps({
        "job_id": "legacy-route-shape",
        "identity": current.spec.identity.as_dict(),
        "route": "old-route-format",
        "status": "legacy-unknown-status",
        "last_exit_code": None,
        "result_dir": "legacy/results",
    }))
    (store.jobs_root / "invalid_identity" / "job_state.json").parent.mkdir(parents=True)
    (store.jobs_root / "invalid_identity" / "job_state.json").write_text(json.dumps({
        "job_id": "invalid-identity-shape",
        "identity": ["not", "a", "mapping"],
        "route": current.spec.route.as_dict(),
        "status": "completed",
        "last_exit_code": 0,
    }))

    store.save(current)

    group = store._compatibility_group(
        current.spec.identity.as_dict(), current.spec.route,
    )
    method = json.loads((group["method_root"] / "method_state.json").read_text())
    task = json.loads((group["task_root"] / "task_state.json").read_text())
    replay = json.loads((group["task_root"] / "replay_manifest.json").read_text())

    assert method["candidate_count"] == 1
    assert task["candidate_count"] == 2
    assert task["counts"]["queued"] == 2
    assert task["family_count"] == 1
    assert {entry["job_id"] for entry in replay["candidates"]} == {
        current.job_id, "legacy-route-shape",
    }
    legacy = next(entry for entry in replay["candidates"] if entry["job_id"] == "legacy-route-shape")
    assert legacy["model_id"] is None


def test_compatibility_views_skip_unhashable_route_identity_fields(tmp_path):
    store = JobStateStore(tmp_path / "run")
    current = _runtime(tmp_path, "well-formed", status=RunStatus.QUEUED)
    malformed_root = store.jobs_root / "malformed_route_values"
    malformed_root.mkdir(parents=True)
    base = {
        "identity": current.spec.identity.as_dict(),
        "status": "unknown-status",
        "last_exit_code": None,
        "result_dir": "malformed/results",
    }
    for name, route in (
        ("list-model-id", {"model_id": ["not", "a", "scalar"]}),
        ("mapping-profile", {"model_id": "OtherModel", "generic_profile": {"bad": "shape"}}),
    ):
        path = malformed_root / name / "job_state.json"
        path.parent.mkdir()
        path.write_text(json.dumps({**base, "job_id": name, "route": route}))

    store.save(current)

    group = store._compatibility_group(
        current.spec.identity.as_dict(), current.spec.route,
    )
    task = json.loads((group["task_root"] / "task_state.json").read_text())
    replay = json.loads((group["task_root"] / "replay_manifest.json").read_text())

    assert task["candidate_count"] == 3
    assert task["counts"]["queued"] == 3
    assert task["family_count"] == 1
    assert task["method_roots"] == [str(group["method_root"])]
    assert {entry["job_id"] for entry in replay["candidates"]} == {
        current.job_id, "list-model-id", "mapping-profile",
    }


def test_invalid_utf8_result_and_status_json_use_empty_fallbacks(tmp_path):
    run_root = tmp_path / "run"
    results_dir = tmp_path / "results"
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="sig", runs=()), RunnerSettings(),
        run_root=run_root, results_dir=results_dir, repository_root=tmp_path,
    )
    runtime = _runtime(tmp_path, "binary-result", status=RunStatus.COMPLETED, exit_code=0)
    result_path = Path(runtime.spec.result_dir) / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(b"\xff\xfe")
    assert coordinator._load_worker_result(runtime) is None

    (run_root / "status.json").write_bytes(b"\xff\xfe")
    summary = coordinator.write_persisted_state_report()
    assert summary["resource_snapshot"] == {}
