from simple_adp.runner.admission import (
    AdmissionState, LaneCounts, choose_device, hard_pressure, parent_cpu_masked,
    pressure_pause_reopened, resolve_launch_capacity, update_soft_gates,
)
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.resources import GPUMemorySample, HostMemorySample
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute, SimpleADPModelAdapter
from simple_adp.runner.plan import RunIdentity, RunSpec, SplitSpec, SuitePlan
from simple_adp.runner.backend import resolve_torch_device
from simple_adp.runner.runtime_tuning import launcher_child_env
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.screening import build_screening_reports
from simple_adp.runner.process_cleanup import owns_process
from simple_adp import MODEL_REGISTRY, RegressionTask, SearchSettings, TabularADP
import numpy as np
import pytest
import torch
import time
import json
import csv
import logging
from collections import deque
from types import SimpleNamespace
from dataclasses import replace
import simple_adp.estimator as estimator_module


def test_emergency_kill_ownership_requires_entrypoint_marker_and_repo_scope(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert owns_process(
        ["python", "-m", "simple_adp.runner.controller", "--plan", "suite.json"],
        str(root / "run"), root,
    )
    assert not owns_process(
        ["python", "-m", "simple_adp.runner.controller"], str(root), root,
    )
    assert not owns_process(
        ["python", "-m", "simple_adp.runner.controller", "--spec", "job.json"],
        str(root), root,
    )
    assert not owns_process(
        ["python", "-m", "simple_adp.runner.worker", "--model", "ADP-W-All"],
        str(root), root,
    )
    assert not owns_process(
        ["python", "-m", "simple_adp.runner.launcher", "--model", "ADP-W-All"],
        str(root), root,
    )
    assert not owns_process(
        ["python", "-m", "unrelated.module", "--plan", "suite.json"], str(root), root,
    )
    assert not owns_process(
        ["python", "-m", "simple_adp.runner.controller", "--plan", "suite.json"],
        str(tmp_path / "outside"), root,
    )
    # An entrypoint name embedded in an unrelated argument is not an owned
    # command, even when it runs from the repository and carries a marker flag.
    assert not owns_process(
        ["python", "-c", "simple_adp.runner.controller", "--plan", "suite.json"],
        str(root), root,
    )
    assert owns_process(
        ["bash", str(root / "scripts" / "simple-adp-suite.sh"), "--output", "results"],
        str(tmp_path / "outside"), root,
    )
    assert owns_process(
        [str(root / "scripts" / "simple-adp-suite.sh"), "--output", "results"],
        str(tmp_path / "outside"), root,
    )
    assert owns_process(
        ["pwsh", "-File", str(root / "scripts" / "simple-adp-suite.ps1"), "--output", "results"],
        str(tmp_path / "outside"), root,
    )
    assert not owns_process(
        ["python", "-m", "py_compile", str(root / "scripts" / "simple-adp-suite.sh"), "--mode"],
        str(root), root,
    )
    assert not owns_process(
        ["python", "-c", "print('ok')", "-m", "simple_adp.runner.worker", "--spec", "job.json"],
        str(root), root,
    )
    assert owns_process(
        ["python", "-m", "simple_adp.runner.worker", "--spec", str(root / "job.json")],
        str(tmp_path / "outside"), root,
    )
    assert owns_process(
        ["python", "-m", "simple_adp.runner.controller", f"--plan={root / 'suite.json'}"],
        str(tmp_path / "outside"), root,
    )
    assert not owns_process(
        ["python", "-m", "simple_adp.runner.controller", "--plan", "."],
        str(tmp_path / "outside"), root,
    )


def test_windows_wrapper_contract_values_and_cleanup_ownership():
    from pathlib import Path

    scripts = Path(__file__).parents[1] / "scripts"
    wrapper = (scripts / "simple-adp-suite.ps1").read_text()
    thermal = (scripts / "simple-adp-thermal-guard.ps1").read_text()
    watchdog = (scripts / "simple-adp-watchdog.ps1").read_text()
    assert "SIMPLE_ADP_ENABLE_THERMAL_GUARD" in wrapper
    assert "$EnableThermal = $true" in wrapper
    assert "if (-not [string]::IsNullOrWhiteSpace($env:SIMPLE_ADP_ENABLE_THERMAL_GUARD)) {" in wrapper
    assert "$EnableThermal = Test-RunnerTrue $env:SIMPLE_ADP_ENABLE_THERMAL_GUARD" in wrapper
    assert "/setactive $PreviousScheme" in wrapper
    assert "$HotTempC = 90" in thermal
    assert "$CoolTempC = 85" in thermal
    assert "$HotCpuMax = 100" in thermal
    assert "$NormalCpuMax = 100" in thermal
    assert "$PollSeconds = 5" in thermal
    assert "Start-Sleep -Seconds $PollSeconds" in thermal
    assert "[Math]::Max(1, $PollSeconds)" not in thermal
    exact_settings = (
        "Subgroup = 'SUB_VIDEO'; Setting = 'VIDEOIDLE'; Value = 5",
        "Subgroup = 'SUB_SLEEP'; Setting = 'STANDBYIDLE'; Value = 0",
        "Subgroup = 'SUB_SLEEP'; Setting = 'HIBERNATEIDLE'; Value = 0",
        "Subgroup = 'SUB_DISK'; Setting = 'DISKIDLE'; Value = 0",
        "Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '29f6c1db-86da-48c5-9fdb-f2b67b1f44da'; Value = 0",
        "Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '9d7815a6-7ee4-497e-8888-515a05f02364'; Value = 0",
        "Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '94ac6d29-73ce-41a6-809f-6363ba21b47e'; Value = 0",
        "Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '7bc4a2f9-d8fc-4469-b07b-33eb785aaca0'; Value = 0",
        "Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '25dfa149-5dd1-4736-b5ab-e8a37b5b8187'; Value = 1",
        "Subgroup = '4f971e89-eebd-4455-a8de-9e59040e7347'; Setting = '5ca83367-6e45-459f-a27b-476b1d01c9360'; Value = 0",
        "Subgroup = '54533251-82be-4824-96c1-47b60b740d00'; Setting = '893dee8e-2bef-41e0-89c6-b55d0929964c'; Value = 5",
        "Subgroup = '54533251-82be-4824-96c1-47b60b740d00'; Setting = 'bc5038f7-23e0-4960-96da-33abaf5935ec'; Value = $NormalCpuMax",
        "Subgroup = '54533251-82be-4824-96c1-47b60b740d00'; Setting = '94d3a615-a899-4ac5-ae2b-e4d8f634367f'; Value = 0",
    )
    setting_positions = []
    for token in exact_settings:
        assert token in thermal
        setting_positions.append(thermal.index(token))
    assert setting_positions == sorted(setting_positions)
    assert "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c" in thermal
    assert thermal.index("root/LibreHardwareMonitor") < thermal.index("root/OpenHardwareMonitor") < thermal.index("root/wmi")
    assert thermal.count("$HotTempC") == 1 and thermal.count("$CoolTempC") == 1
    assert thermal.count("$HotCpuMax") == 1  # recorded threshold knobs are unused
    assert thermal.count("$NormalCpuMax") == 2
    assert "SetThreadExecutionState($Continuous -bor $SystemRequired -bor $AwayModeRequired)" in thermal
    assert "SetThreadExecutionState($Continuous)" in thermal
    assert "if ($null -ne $temperature) { Write-GuardLog" in thermal
    assert "SetThreadExecutionState($Continuous -bor $SystemRequired -bor $AwayModeRequired)" in thermal
    assert "SetThreadExecutionState($Continuous)" in thermal
    assert "$OriginalScheme" not in thermal
    assert "simple-adp-watchdog.ps1" in wrapper


def test_windows_wrapper_uses_parent_watchdog_event_and_optional_awake_helper():
    from pathlib import Path

    scripts = Path(__file__).parents[1] / "scripts"
    wrapper = (scripts / "simple-adp-suite.ps1").read_text()
    watchdog = (scripts / "simple-adp-watchdog.ps1").read_text()
    assert "Start-SimpleADPMasterWatchdog -MasterPid $PID" in wrapper
    assert "Stop-SimpleADPMasterWatchdog -WatchdogJob" in wrapper
    assert "strong_awake_guard.ps1" in wrapper
    assert "Register-EngineEvent" in watchdog
    assert "Unregister-Event" in watchdog
    assert "Start-Sleep -Seconds 3" in watchdog
    assert "GraceSeconds = 15" in watchdog
    assert "PollSeconds = 3" in watchdog
    assert "Sort-Object -Descending" in watchdog
    cleanup = wrapper[wrapper.index("} finally {"):]
    assert "if ($EnableThermal -and $StopFile)" in cleanup
    assert cleanup.index("if ($EnableThermal -and $StopFile)") < cleanup.index("$ThermalProcess.Id")
    assert cleanup.index("$StopFile -Value") < cleanup.index("$ThermalProcess.Id")
    assert cleanup.index("$ThermalProcess.Id") < cleanup.index("$AwakeProcess.Id")
    assert cleanup.index("$AwakeProcess.Id") < cleanup.index("Stop-SimpleADPMasterWatchdog")
    assert cleanup.index("Stop-SimpleADPMasterWatchdog") < cleanup.index("$LauncherProcess.Id")
    assert cleanup.index("$LauncherProcess.Id") < cleanup.index("powercfg /setactive")
    assert cleanup.index("powercfg /setactive") < cleanup.index("$StopFile -Force")


def test_latest_attempt_log_scan_uses_last_marker_and_handles_chunk_boundary(tmp_path):
    from simple_adp.runner.scheduler import SuiteCoordinator

    marker = b"=== LAUNCH 2 ===\n"
    path = tmp_path / "attempts.log"
    path.write_bytes(b"x" * 10_000 + marker + b"y" * 65_524)
    latest = SuiteCoordinator._latest_attempt_text(str(path))
    assert latest.startswith("=== LAUNCH 2 ===")
    assert "y" in latest
    assert "x" not in latest


def test_latest_attempt_log_backward_scan_finds_marker_beyond_fallback_window(tmp_path):
    from simple_adp.runner.scheduler import SuiteCoordinator

    path = tmp_path / "large-attempts.log"
    path.write_bytes(
        b"=== LAUNCH old ===\n" + b"old-attempt\n" + b"x" * 4096
        + b"=== LAUNCH current ===\n" + b"current-attempt\n" + b"y" * 450_000
    )

    latest = SuiteCoordinator._latest_attempt_text(str(path))

    assert latest.startswith("=== LAUNCH current ===")
    assert "old-attempt" not in latest
    assert latest.count("current-attempt") == 1
    assert len(latest.encode("utf-8")) > 200_000


def test_recovery_reconciliation_classifies_terminal_paused_and_unknown_states(tmp_path):
    from simple_adp.runner.processes import reconcile_runtime
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="probe", seed=4,
        split_sha256="split", run_signature="suite-signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("__simple_adp_missing_test_command__",))

    def runtime():
        return JobRuntime(spec)

    completed = runtime()
    assert reconcile_runtime(completed, {"status": "completed", "exit_code": 0}) == "completed"
    assert completed.status == RunStatus.COMPLETED
    paused = runtime()
    assert reconcile_runtime(paused, {
        "status": "paused", "pause_reason": "host_ram_pressure",
        "gpu_pause_count": "2", "host_pause_count": 3,
        "force_cpu_after_cuda_oom": True,
    }) == "suspended"
    assert paused.status == RunStatus.PAUSED
    assert (paused.gpu_pause_count, paused.host_pause_count) == (2, 3)
    assert paused.force_cpu_after_cuda_oom is False
    terminal = runtime()
    assert reconcile_runtime(terminal, {"status": "failed", "last_error": "inexact_checkpoint_resume"}) == "terminal"
    queued = runtime()
    assert reconcile_runtime(queued, {"status": "running", "pid": 999999999}) == "queued"
    assert queued.status == RunStatus.QUEUED


def test_startup_queues_failed_job_even_when_persisted_retry_budget_is_exhausted(tmp_path):
    """Retry limits apply after a new failure, not before startup reconciliation."""
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="exhausted-retry", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="exhausted-retry", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "exhausted-retry"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"), {},
        ("python", "worker.py"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))
    settings = RunnerSettings(max_retries=1, run_hardness_screen=False)
    coordinator = SuiteCoordinator(
        plan, settings, run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    persisted = JobRuntime(
        spec, status=RunStatus.FAILED, attempt=2, retry_count=1,
        exit_code=9, last_exit_code=9, last_error="nonzero_exit:9",
    )
    coordinator.state_store.save(persisted)

    active, suspended, queue = coordinator._build_runtimes()

    assert active == [] and suspended == []
    assert len(queue) == 1
    recovered = queue[0]
    assert recovered.status == RunStatus.QUEUED
    assert recovered.attempt == 2 and recovered.retry_count == settings.max_retries
    assert coordinator._should_retry(settings, recovered) is False


@pytest.fixture(scope="module")
def _shared_exit_matrix_root(tmp_path_factory):
    return tmp_path_factory.mktemp("shared-exit-pressure-matrix")


@pytest.mark.parametrize(
    ("status", "reason_field"),
    [("failed", "last_error"), ("fail", "reason")],
)
def test_exact_resume_terminal_failure_precedes_live_process_recovery(
    tmp_path, monkeypatch, status, reason_field,
):
    import simple_adp.runner.processes as processes
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="terminal-live", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="terminal-live", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "terminal-live"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {},
        ("python", "worker.py"),
    )
    monkeypatch.setattr(processes, "process_is_live", lambda pid: pid == 818)
    monkeypatch.setattr(
        processes, "find_exact_argv",
        lambda _command: (_ for _ in ()).throw(AssertionError("terminal state must short-circuit")),
    )
    runtime = JobRuntime(spec)
    record = {
        "status": status, "pid": 818,
        reason_field: "inexact_checkpoint_resume",
    }

    assert processes.reconcile_runtime(runtime, record) == "terminal"
    assert runtime.status == RunStatus.FAILED
    assert runtime.process is None


def test_recovery_reconciliation_safely_defaults_malformed_persisted_fields(tmp_path, monkeypatch):
    import simple_adp.runner.processes as processes
    from simple_adp.runner.processes import reconcile_runtime
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="malformed", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="probe", seed=4,
        split_sha256="split", run_signature="suite-signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "malformed"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {},
        ("__simple_adp_missing_test_command__",),
    )
    monkeypatch.setattr(processes, "find_exact_argv", lambda _command: None)
    runtime = JobRuntime(spec)

    location = reconcile_runtime(runtime, {
        "status": "running", "attempt": "bad", "retry_count": None,
        "pid": "not-a-pid", "pause_count": object(),
        "started_at": "invalid", "ended_at": None,
        "launch_context": ["not", "a", "mapping"],
        "resource_snapshot": None,
    })

    assert location == "queued"
    assert runtime.status == RunStatus.QUEUED
    assert (runtime.attempt, runtime.retry_count, runtime.pid, runtime.pause_count) == (0, 0, None, 0)
    assert (runtime.started_at, runtime.ended_at) == (0.0, None)
    assert runtime.launch_context == {}
    assert runtime.resource_snapshot == {}


@pytest.mark.parametrize("exit_code", [0, None, "0"])
def test_completed_recovery_record_is_not_requeued_from_a_stale_live_pid(tmp_path, monkeypatch, exit_code):
    import simple_adp.runner.processes as processes
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="completed-live", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="completed-live", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "completed-live"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("python", "worker.py"),
    )
    monkeypatch.setattr(processes, "process_is_live", lambda _pid: True)
    monkeypatch.setattr(
        processes, "find_exact_argv",
        lambda _command: (_ for _ in ()).throw(AssertionError("live persisted PID takes precedence")),
    )
    runtime = JobRuntime(spec)

    location = processes.reconcile_runtime(runtime, {
        "status": "completed", "exit_code": exit_code,
        "last_exit_code": exit_code, "pid": 4321,
        "ended_at": 12.0,
    })

    assert location == "completed"
    assert runtime.status == RunStatus.COMPLETED
    assert runtime.exit_code == 0
    assert runtime.process is None
    assert runtime.recovered_process is False


@pytest.mark.parametrize(
    ("exit_code", "last_exit_code", "expected"),
    [(2, 0, "completed"), (0, 2, "queued")],
)
def test_completed_reconciliation_uses_source_last_exit_code(
    tmp_path, monkeypatch, exit_code, last_exit_code, expected,
):
    import simple_adp.runner.processes as processes
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="exit-fields", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="exit-fields", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "exit-fields"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {},
        ("__simple_adp_missing_test_command__",),
    )
    monkeypatch.setattr(processes, "process_is_live", lambda _pid: False)
    monkeypatch.setattr(processes, "find_exact_argv", lambda _command: None)
    runtime = JobRuntime(spec)
    location = processes.reconcile_runtime(runtime, {
        "status": "completed", "exit_code": exit_code,
        "last_exit_code": last_exit_code,
    })

    assert location == expected
    assert runtime.status == (RunStatus.COMPLETED if expected == "completed" else RunStatus.QUEUED)


@pytest.mark.parametrize(
    ("status", "saved_pid", "live_pid", "matched_pid", "persist"),
    [
        ("launching", 777, 777, None, True),
        ("retrying", None, None, 777, True),
        ("running", 777, 777, None, False),
    ],
)
def test_live_launch_recovery_persists_normalization_but_same_running_pid_is_unchanged(
    tmp_path, monkeypatch, status, saved_pid, live_pid, matched_pid, persist,
):
    import re
    import simple_adp.runner.processes as processes
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="recover-live", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="recover-live", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "recover-live"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("python", "worker.py"),
    )
    monkeypatch.setattr(processes, "process_is_live", lambda pid: pid == live_pid and pid is not None)
    matched = SimpleNamespace(pid=matched_pid) if matched_pid is not None else None
    monkeypatch.setattr(processes, "find_exact_argv", lambda _command: matched)
    saved = []

    class Store:
        def save(self, runtime, *, updated_at=None):
            record = runtime.to_record()
            if updated_at is not None:
                record["updated_at"] = updated_at
            saved.append(record)

    runtime = JobRuntime(spec)
    location = processes.reconcile_runtime(
        runtime,
        {
            "status": status, "attempt": 2, "retry_count": 1,
            "pid": saved_pid, "device_mode": "gpu", "pause_reason": "stale",
            "exit_code": 9, "started_at": 10.0, "ended_at": 20.0,
        },
        state_store=Store(),
    )

    assert location == "active"
    assert runtime.status == RunStatus.RUNNING
    assert runtime.pid == 777
    assert (len(saved) == 1) is persist
    if persist:
        state = saved[0]
        assert state["status"] == "running"
        assert state["pid"] == 777
        assert state["pause_reason"] == ""
        assert state["exit_code"] is None and state["ended_at"] is None
        assert state["recovered_from_status"] == status
        assert re.fullmatch(r"\d{8}_\d{6}", state["recovered_at"])
        assert state["updated_at"] == state["recovered_at"]


def test_live_launch_recovery_round_trips_reconciled_state_to_disk(tmp_path, monkeypatch):
    import re
    import simple_adp.runner.processes as processes
    from simple_adp.runner.state import JobRuntime, JobStateStore, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="recover-disk", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="recover-disk", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "recover-disk"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "results"), {}, ("python", "worker.py"),
    )
    store = JobStateStore(tmp_path / "run")
    initial = JobRuntime(
        spec, status=RunStatus.LAUNCHING, attempt=2, retry_count=1, pid=888,
        device_mode="gpu", pause_reason="stale", exit_code=9,
        started_at=10.0, ended_at=20.0,
    )
    store.save(initial)
    original_record = store.load(initial.job_id)
    original_record["future_state_field"] = {"version": 2, "enabled": True}
    store.state_path(initial).write_text(json.dumps(original_record), encoding="utf-8")
    monkeypatch.setattr(processes, "process_is_live", lambda pid: pid == 888)
    monkeypatch.setattr(
        processes, "find_exact_argv",
        lambda _command: (_ for _ in ()).throw(AssertionError("persisted live PID should be used")),
    )
    recovered = JobRuntime(spec)

    location = processes.reconcile_runtime(
        recovered, store.load(recovered.job_id), state_store=store,
    )
    saved = store.load(recovered.job_id)

    assert location == "active"
    assert saved["status"] == "running"
    assert saved["pid"] == 888
    assert saved["pause_reason"] == ""
    assert saved["exit_code"] is None and saved["ended_at"] is None
    assert saved["recovered_from_status"] == "launching"
    assert saved["future_state_field"] == {"version": 2, "enabled": True}
    assert re.fullmatch(r"\d{8}_\d{6}", saved["recovered_at"])
    assert saved["updated_at"] == saved["recovered_at"]


@pytest.mark.parametrize("persisted_status", ["queued", "failed"])
def test_live_saved_pid_overrides_nonterminal_persisted_status(
    tmp_path, monkeypatch, persisted_status,
):
    import simple_adp.runner.processes as processes
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id=f"live-{persisted_status}",
        split_repeat=0, fold=0, sample=0, suite_repeat=1,
        model_route=f"live-{persisted_status}", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", identity.task_id), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {},
        ("python", "worker.py", "--job", identity.task_id),
    )
    monkeypatch.setattr(processes, "process_is_live", lambda pid: pid == 717)
    monkeypatch.setattr(
        processes, "find_exact_argv",
        lambda _command: (_ for _ in ()).throw(AssertionError("saved live PID takes precedence")),
    )

    runtime = JobRuntime(spec)
    location = processes.reconcile_runtime(runtime, {
        "status": persisted_status, "pid": 717, "device_mode": "cpu",
    })

    assert location == "active"
    assert runtime.status == RunStatus.RUNNING
    assert runtime.pid == 717
    assert runtime.recovered_process is True


def test_recovered_process_handle_reports_synthetic_exit_after_pid_disappears(monkeypatch):
    import simple_adp.runner.processes as processes

    handle = processes.RecoveredProcessHandle(515)
    monkeypatch.setattr(processes, "process_is_live", lambda pid: pid == 515)
    assert handle.poll() is None
    monkeypatch.setattr(processes, "process_is_live", lambda _pid: False)
    assert handle.poll() == 1


def test_runtime_derived_status_priority_and_paused_live_recovery(tmp_path, monkeypatch):
    import simple_adp.runner.processes as processes
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="status-priority", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("python",),
    )
    completed_live = JobRuntime(
        spec, status=RunStatus.PAUSED, exit_code=0, last_exit_code=0, ended_at=1.0,
        process=SimpleNamespace(poll=lambda: None),
    )
    assert completed_live.derived_status == RunStatus.COMPLETED
    live_over_paused = JobRuntime(
        spec, status=RunStatus.PAUSED, process=SimpleNamespace(poll=lambda: None),
    )
    assert live_over_paused.derived_status == RunStatus.RUNNING
    assert JobRuntime(spec, status=RunStatus.PAUSED).derived_status == RunStatus.PAUSED
    assert JobRuntime(spec).derived_status == RunStatus.QUEUED
    assert JobRuntime(spec, exit_code=3, last_exit_code=3).derived_status == RunStatus.FAILED
    # The source property's last branch classifies a retained zero exit as
    # completed even when its end timestamp is empty.
    assert JobRuntime(spec, exit_code=0, last_exit_code=0).derived_status == RunStatus.COMPLETED
    # A retained source exit code controls derived status when current-attempt
    # state conflicts with it.
    assert JobRuntime(spec, exit_code=0, last_exit_code=7, ended_at=1.0).derived_status == RunStatus.FAILED
    assert JobRuntime(spec, exit_code=7, last_exit_code=0).derived_status == RunStatus.COMPLETED
    assert JobRuntime(spec, exit_code=7, last_exit_code=None).derived_status == RunStatus.QUEUED

    monkeypatch.setattr(processes, "process_is_live", lambda pid: pid == 717)
    monkeypatch.setattr(processes, "find_exact_argv", lambda _command: SimpleNamespace(pid=717))
    recovered = JobRuntime(spec)
    location = processes.reconcile_runtime(recovered, {
        "status": "paused", "pid": 717, "device_mode": "cpu",
    })
    assert location == "active"
    assert recovered.status == RunStatus.RUNNING
    assert recovered.recovered_process is True


def test_paused_recovery_discards_live_saved_pid_without_exact_command_match(
    tmp_path, monkeypatch,
):
    import simple_adp.runner.processes as processes
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="paused-pid-reuse", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="paused-pid-reuse", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "paused-pid-reuse"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {},
        ("python", "worker.py", "--job", "original"),
    )
    monkeypatch.setattr(processes, "process_is_live", lambda pid: pid == 717)
    monkeypatch.setattr(processes, "find_exact_argv", lambda _command: None)

    store = processes.JobStateStore(tmp_path / "run")

    recovered = JobRuntime(spec)
    location = processes.reconcile_runtime(recovered, {
        "status": "paused", "pid": 717, "device_mode": "cpu",
        "pause_reason": "host_ram_pressure",
        "last_exit_code": 9, "ended_at": 12.0,
    }, state_store=store)

    assert location == "suspended"
    assert recovered.status == RunStatus.PAUSED
    assert recovered.pid is None
    assert recovered.process is None
    assert recovered.recovered_process is False
    saved_record = store.load(recovered.job_id)
    assert saved_record["status"] == "paused"
    assert saved_record["pid"] is None
    assert saved_record["last_exit_code"] is None
    assert saved_record["ended_at"] is None
    assert saved_record["recovered_from_status"] == "paused"
    assert saved_record["recovered_at"] == saved_record["updated_at"]
    assert not (store.run_root / "candidate_state.json").exists()


def test_device_capable_job_metadata_round_trips_and_defaults_true(tmp_path):
    import json

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="device-metadata", seed=4,
        split_sha256="split", run_signature="signature",
    )
    base = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("python",),
    )
    assert base.device_capable is True
    assert base.as_dict()["device_capable"] is True

    disabled = RunSpec(
        identity, base.task_ref, base.model_adapter, base.route, base.split_file,
        base.result_dir, base.worker_settings, base.command, device_capable=False,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "run_signature": "signature", "jobs": [disabled.as_dict()],
    }))
    loaded = SuitePlan.from_manifest(manifest_path)
    assert loaded.runs[0].device_capable is False

    legacy_job = disabled.as_dict()
    legacy_job.pop("device_capable")
    manifest_path.write_text(json.dumps({"run_signature": "signature", "jobs": [legacy_job]}))
    assert SuitePlan.from_manifest(manifest_path).runs[0].device_capable is True


def test_pressure_pause_updates_total_and_reason_specific_counters(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="dual-pause", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("python",))
    runtime = JobRuntime(
        spec, status=RunStatus.RUNNING, device_mode="gpu",
        exit_code=7, last_exit_code=7, ended_at=12.0,
    )
    runtime.pid = 515
    runtime.process = SimpleNamespace(pid=515, poll=lambda: None)
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="signature", runs=(spec,)), RunnerSettings(),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = SimpleNamespace(as_dict=lambda: {"sample": "pressure"})
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)

    assert coordinator._pause(runtime, [runtime], [], "gpu_memory_pressure+host_ram_pressure", snapshot)
    assert (runtime.pause_count, runtime.gpu_pause_count, runtime.host_pause_count) == (1, 1, 1)
    assert runtime.exit_code is None and runtime.last_exit_code is None and runtime.ended_at is None
    paused_state = coordinator.state_store.load(runtime.job_id)
    assert paused_state["exit_code"] is None
    assert paused_state["last_exit_code"] is None
    assert paused_state["ended_at"] is None


def test_gpu_oom_retry_increments_legacy_gpu_pause_counter(tmp_path, monkeypatch):
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="oom-route", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("python",))
    runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode="gpu")
    runtime.process = SimpleNamespace(poll=lambda: 9)
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="signature", runs=(spec,)), RunnerSettings(max_retries=1),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    coordinator.memory.values["last_backoff_ts"] = "prior_timestamp"
    monkeypatch.setattr(coordinator, "_classify_failure", lambda *_args: "cuda_oom")
    snapshot = SimpleNamespace(gpu=SimpleNamespace(used_mib=7000), as_dict=lambda: {})
    queue = deque()

    coordinator._handle_exit(runtime, [runtime], [], queue, snapshot)

    assert runtime.gpu_pause_count == 1
    assert runtime.force_cpu_after_cuda_oom is False
    assert coordinator.admission.batch_scale == 0.5
    assert coordinator.admission.last_backoff_reason == "cuda_oom"
    assert time.strptime(coordinator.memory.values["last_backoff_ts"], "%Y%m%d_%H%M%S")
    persisted_backoff = json.loads(coordinator.batch_state_path.read_text(encoding="utf-8"))
    assert persisted_backoff["batch_scale"] == 0.5
    assert persisted_backoff["last_backoff_ts"] == coordinator.memory.values["last_backoff_ts"]
    assert list(queue) == [runtime]


def test_full_pass_removes_active_runtime_with_missing_process_handle(tmp_path, monkeypatch):
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="missing-process", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="missing-process", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "missing-process")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(tmp_path / "result"), {}, ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))
    runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode="cpu", process=None)
    active = [runtime]
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(cpu_only=True, concurrency=1, post_launch_sample_delay_seconds=0),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(
        _host(), None, 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 1,
    )
    sample_calls = []

    def sample(*_args):
        sample_calls.append(1)
        if len(sample_calls) > 1:
            raise AssertionError("missing-process runtime must be removed in the same pass")
        return snapshot, None, None

    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: (active, [], deque()))
    monkeypatch.setattr(coordinator, "_sample_snapshot", sample)
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    result = coordinator.run()

    assert result["run_signature"] == "signature"
    assert active == []
    assert runtime.status is RunStatus.RUNNING  # source drops the handle-less entry without finalizing it
    assert len(sample_calls) == 1


def test_cuda_oom_at_backoff_floor_preserves_previous_backoff_record(tmp_path, monkeypatch):
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="oom-floor", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("python",),
    )
    runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode="gpu")
    runtime.process = SimpleNamespace(poll=lambda: 9)
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="signature", runs=(spec,)),
        RunnerSettings(max_retries=1), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    coordinator.admission.batch_scale = 1e-6
    coordinator.admission.last_backoff_reason = "prior_oom"
    coordinator.memory.values["last_backoff_ts"] = "prior_timestamp"
    monkeypatch.setattr(coordinator, "_classify_failure", lambda *_args: "cuda_oom")

    coordinator._handle_exit(
        runtime, [runtime], [], deque(),
        SimpleNamespace(gpu=None, as_dict=lambda: {}),
    )

    assert coordinator.admission.batch_scale == 1e-6
    assert coordinator.admission.last_backoff_reason == "prior_oom"
    assert coordinator.memory.values["last_backoff_ts"] == "prior_timestamp"


@pytest.mark.parametrize("device_mode", ["cpu", "gpu"])
def test_non_oom_retry_releases_device_gates_and_launch_flags(tmp_path, monkeypatch, device_mode, caplog):
    import logging
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route=f"ordinary-failure-{device_mode}", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("python",),
    )
    runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode=device_mode)
    runtime.process = SimpleNamespace(poll=lambda: 1)
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="signature", runs=(spec,)),
        RunnerSettings(max_retries=1), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    coordinator.memory.values["last_backoff_ts"] = "prior_timestamp"
    coordinator.admission.gpu_gate_closed = True
    coordinator.admission.host_gate_closed = True
    coordinator.admission.gpu_gate_lock_job_id = "gpu-lock"
    coordinator.admission.host_gate_lock_job_id = "host-lock"
    coordinator.admission.gpu_launches_enabled = False
    coordinator.admission.cpu_launches_enabled = False
    monkeypatch.setattr(coordinator, "_classify_failure", lambda *_args: "nonzero_exit:1")

    with caplog.at_level(logging.INFO, logger="simple_adp.runner"):
        coordinator._handle_exit(
            runtime, [runtime], [], deque(),
            SimpleNamespace(gpu=None, as_dict=lambda: {}),
        )

    assert runtime.status == RunStatus.RETRYING
    # The ledger's device-mode release helper clears gate locks and re-enables
    # the affected launch lanes when a retry is queued.
    assert coordinator.admission.host_gate_closed is False
    assert coordinator.admission.host_gate_lock_job_id is None
    assert coordinator.admission.cpu_launches_enabled is True
    assert coordinator.admission.gpu_gate_closed is (device_mode != "gpu")
    assert coordinator.admission.gpu_gate_lock_job_id == (None if device_mode == "gpu" else "gpu-lock")
    assert coordinator.admission.gpu_launches_enabled is (device_mode == "gpu")
    assert coordinator.admission.last_backoff_reason is None
    assert coordinator.memory.values["last_backoff_ts"] is None
    persisted_backoff = json.loads(coordinator.batch_state_path.read_text(encoding="utf-8"))
    assert persisted_backoff["last_backoff_ts"] is None
    messages = [record.getMessage() for record in caplog.records]
    released = "gpu,host" if device_mode == "gpu" else "host"
    release_message = f"[GATE] admission reopened job={runtime.job_id} released={released}"
    retry_message = next(message for message in messages if message.startswith("[RETRY]"))
    assert messages.index(release_message) < messages.index(retry_message)


@pytest.mark.parametrize(
    ("failure_at", "expected_calls"),
    [("write", ["write"]), ("flush", ["write", "flush"])],
)
def test_retry_log_cleanup_preserves_inline_source_exception_boundary(
    tmp_path, monkeypatch, failure_at, expected_calls,
):
    from simple_adp.runner.state import JobRuntime, RunStatus

    class FaultingLog:
        def __init__(self):
            self.calls = []

        def write(self, _value):
            self.calls.append("write")
            if failure_at == "write":
                raise OSError("write failed")

        def flush(self):
            self.calls.append("flush")
            if failure_at == "flush":
                raise OSError("flush failed")

        def close(self):
            self.calls.append("close")

    spec = RunSpec(
        RunIdentity("toy", "suite", "retry-log", 0, 0, 0, 1, "retry-log", 4, "split", "signature"),
        DatasetTaskRef("toy", "suite", "retry-log"), "probe", ModelRoute("ADP-W-All"),
        "split.npz", str(tmp_path / "result"), {}, ("python", "worker.py"),
    )
    handle = FaultingLog()
    runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode="cpu", log_handle=handle)
    runtime.process = SimpleNamespace(poll=lambda: 1)
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="signature", runs=(spec,)),
        RunnerSettings(max_retries=1), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    monkeypatch.setattr(coordinator, "_classify_failure", lambda *_args: "nonzero_exit:1")

    coordinator._handle_exit(
        runtime, [runtime], [], deque(),
        SimpleNamespace(gpu=None, as_dict=lambda: {}),
    )

    assert handle.calls == expected_calls
    assert runtime.log_handle is None


def test_terminal_exact_resume_failure_persists_before_gate_release(tmp_path, monkeypatch):
    from simple_adp.runner.state import JobRuntime, RunStatus

    spec = RunSpec(
        RunIdentity("toy", "suite", "terminal-order", 0, 0, 0, 1, "terminal-order", 4, "split", "signature"),
        DatasetTaskRef("toy", "suite", "terminal-order"), "probe", ModelRoute("ADP-W-All"),
        "split.npz", str(tmp_path / "result"), {}, ("python", "worker.py"),
    )
    runtime = JobRuntime(spec, status=RunStatus.RUNNING, attempt=1, device_mode="gpu")
    runtime.process = SimpleNamespace(pid=123, poll=lambda: 9)
    coordinator = SuiteCoordinator(
        SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,)),
        RunnerSettings(max_retries=1, run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    coordinator.admission.gpu_gate_closed = True
    coordinator.admission.host_gate_closed = True
    coordinator.admission.gpu_launches_enabled = False
    coordinator.admission.cpu_launches_enabled = False
    observed = []
    monkeypatch.setattr(coordinator, "_classify_failure", lambda *_args: "inexact_checkpoint_resume")
    monkeypatch.setattr(
        coordinator, "_finalize",
        lambda *_args, **_kwargs: observed.append((
            coordinator.admission.gpu_gate_closed,
            coordinator.admission.host_gate_closed,
            coordinator.admission.gpu_launches_enabled,
            coordinator.admission.cpu_launches_enabled,
        )),
    )

    coordinator._handle_exit(runtime, [runtime], [], deque(), SimpleNamespace(as_dict=lambda: {}))

    assert observed == [(True, True, False, False)]
    assert coordinator.admission.gpu_gate_closed is False
    assert coordinator.admission.host_gate_closed is False
    assert coordinator.admission.gpu_launches_enabled is True
    assert coordinator.admission.cpu_launches_enabled is True


@pytest.mark.parametrize("device_mode", ["cpu", "gpu"])
def test_cuda_oom_updates_gpu_baseline_only_for_gpu_job_with_sample(tmp_path, monkeypatch, device_mode):
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route=f"oom-{device_mode}", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz", str(tmp_path), {}, ("python",))
    runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode=device_mode)
    runtime.process = SimpleNamespace(poll=lambda: 9)
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="signature", runs=(spec,)), RunnerSettings(max_retries=1),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    coordinator.memory.values["gpu_peak_vram_mib"] = 321.0
    monkeypatch.setattr(coordinator, "_classify_failure", lambda *_args: "cuda_oom")
    gpu_sample = None if device_mode == "gpu" else SimpleNamespace(used_mib=999.0)
    snapshot = SimpleNamespace(gpu=gpu_sample, as_dict=lambda: {})
    queue = deque()

    coordinator._handle_exit(runtime, [runtime], [], queue, snapshot)

    assert runtime.gpu_pause_count == int(device_mode == "gpu")
    assert coordinator.memory.values["gpu_peak_vram_mib"] == 321.0
    assert list(queue) == [runtime]


def test_full_manifest_digest_allows_only_interpreter_symlink_aliases(tmp_path):
    import json
    import sys
    from pathlib import Path
    from simple_adp.runner.plan import manifest_compatibility_digest

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="probe", seed=4,
        split_sha256="split", run_signature="suite-signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(tmp_path / "result"), {"batch_size": 8},
        (sys.executable, "-m", "simple_adp.runner.worker"),
    )
    plan = SuitePlan("suite-signature", {
        "run_signature": "suite-signature", "python_executable": sys.executable,
        "jobs": [spec.as_dict()],
    }, (spec,))
    run_root = tmp_path / "run"
    manifest_path = plan.write_manifest(run_root)
    alias = tmp_path / "python-alias"
    alias.symlink_to(sys.executable)

    # Only argv[0] participates in source-compatible interpreter aliasing;
    # unrelated manifest fields retain exact string identity.
    executable_manifest = {"python_executable": sys.executable, "jobs": []}
    alias_manifest = {"python_executable": str(alias), "jobs": []}
    assert manifest_compatibility_digest(executable_manifest) != manifest_compatibility_digest(alias_manifest)

    payload = json.loads(manifest_path.read_text())
    payload["jobs"][0]["command"][0] = str(alias)
    manifest_path.write_text(json.dumps(payload))
    plan.write_manifest(run_root)  # absolute interpreter symlink aliases normalize

    payload = json.loads(manifest_path.read_text())
    payload["jobs"][0]["worker_settings"]["batch_size"] = 16
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match"):
        plan.write_manifest(run_root)
    with pytest.raises(ValueError, match="digest is invalid"):
        SuitePlan.from_manifest(manifest_path)


def test_unreadable_existing_manifest_becomes_a_persisted_mismatch(tmp_path):
    plan = SuitePlan("sig", {"run_signature": "sig", "jobs": []}, ())
    run_root = tmp_path / "run"
    manifest_path = plan.write_manifest(run_root)
    manifest_path.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="run manifest does not match"):
        plan.write_manifest(run_root)
    mismatch = json.loads((run_root / "manifest_mismatch.json").read_text(encoding="utf-8"))
    assert mismatch["expected_signature"] == "sig"
    assert mismatch["found_signature"] is None


def test_failed_pressure_resume_stays_paused_and_records_error(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="probe", seed=4,
        split_sha256="split", run_signature="suite-signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"), {}, ("python", "-c", "pass"))
    plan = SuitePlan("suite-signature", {"run_signature": "suite-signature", "jobs": []}, (spec,))
    coordinator = SuiteCoordinator(plan, RunnerSettings(), run_root=tmp_path / "run", results_dir=tmp_path / "result", repository_root=tmp_path)
    runtime = JobRuntime(spec, status=RunStatus.PAUSED, device_mode="gpu", pause_reason="gpu_memory_pressure")
    snapshot = ResourceSnapshot(_host(), _gpu(), 0, 0, 0, 0, 0, 0, 0, 0, frozenset(), frozenset(), False, False, 1)
    monkeypatch.setattr(scheduler_module, "launch_job", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cannot restart")))
    active = []

    assert coordinator._resume(runtime, active, snapshot, 1) is False
    assert runtime.status == RunStatus.PAUSED
    assert runtime.pause_reason == "gpu_memory_pressure"
    assert runtime.last_error == "resume_launch_error"
    assert not active
    assert "resume_launch_error" in (coordinator.state_store.events_path).read_text()


def test_full_pass_resume_launch_failure_keeps_pause_and_still_admits_queued_route(tmp_path, monkeypatch, caplog):
    import logging
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.state import JobRuntime, RunStatus

    def make_spec(label):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0,
            fold=0, sample=0, suite_repeat=1, model_route=label, seed=11,
            split_sha256="split", run_signature="resume-failure-signature",
        )
        return RunSpec(
            identity, DatasetTaskRef("toy", "suite", "1"), "probe",
            ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "results" / label),
            {}, ("python", "-c", "pass"),
        )

    paused = JobRuntime(
        make_spec("paused-route"), status=RunStatus.PAUSED,
        device_mode="gpu", pause_reason="gpu_memory_pressure",
    )
    queued = JobRuntime(make_spec("queued-route"))
    plan = SuitePlan(
        "resume-failure-signature",
        {"run_signature": "resume-failure-signature", "jobs": []},
        (paused.spec, queued.spec),
    )
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(concurrency=2, post_launch_sample_delay_seconds=0, run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = SimpleNamespace(
        host=_host(8192), gpu=_gpu(8192), host_pressure=False,
        gpu_pressure=False, launch_capacity=2, as_dict=lambda: {"sample": "ready"},
    )
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([], [paused], deque([queued])))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_resume_ready", lambda *_args: True)
    monkeypatch.setattr(coordinator, "_suspend_pressure_victim", lambda *_args: False)

    launches = []

    class LiveChild:
        pid = 918

        def poll(self):
            return None

    def launch(runtime, **kwargs):
        resumed = bool(kwargs.get("launch_context", {}).get("resumed_from_pause"))
        launches.append((runtime.job_id, resumed))
        if resumed:
            raise OSError("resume process could not start")
        runtime.process = LiveChild()
        runtime.pid = runtime.process.pid
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = kwargs["device"]
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    coordinator.sleep = lambda _seconds: setattr(coordinator, "shutdown_signal", int(signal.SIGTERM))

    with caplog.at_level(logging.WARNING, logger="simple_adp.runner"):
        with pytest.raises(SystemExit) as stopped:
            coordinator.run()

    assert stopped.value.code == 130
    assert launches == [(paused.job_id, True), (queued.job_id, False)]
    assert paused.status == RunStatus.PAUSED
    assert paused.pause_reason == "gpu_memory_pressure"
    assert queued.status == RunStatus.QUEUED
    assert "[INTERRUPT] Shutdown requested by SIGTERM. Terminating active and suspended jobs..." in [
        record.getMessage() for record in caplog.records
    ]
    events = list(csv.DictReader(coordinator.state_store.events_path.open(newline="", encoding="utf-8")))
    assert [event["event"] for event in events if event["event"] in {"resume_launch_error", "launch"}] == [
        "resume_launch_error", "launch",
    ]
    paused_state = coordinator.state_store.load(paused.job_id)
    queued_state = coordinator.state_store.load(queued.job_id)
    assert paused_state["status"] == "paused"
    assert paused_state["pause_reason"] == "gpu_memory_pressure"
    assert queued_state["status"] == "queued"


def test_pressure_resume_bypasses_capacity_caps_gates_and_holds(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="paused", seed=4,
        split_sha256="split", run_signature="suite-signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(tmp_path / "result"), {}, ("python", "-c", "pass"),
    )
    plan = SuitePlan("suite-signature", {"run_signature": "suite-signature", "jobs": []}, (spec,))
    settings = RunnerSettings(
        concurrency=1, max_active_jobs=1, max_active_gpu_jobs=1,
        post_launch_sample_delay_seconds=30,
    )
    coordinator = SuiteCoordinator(
        plan, settings, run_root=tmp_path / "run", results_dir=tmp_path / "result",
        repository_root=tmp_path,
    )
    coordinator.admission.gpu_gate_closed = True
    coordinator.admission.host_gate_closed = True
    coordinator.admission.gpu_hold_until = 999
    coordinator.admission.cpu_hold_until = 999
    paused = JobRuntime(
        spec, status=RunStatus.PAUSED, device_mode="gpu",
        pause_reason="gpu_memory_pressure",
    )
    paused.pid = 1111
    paused.process = SimpleNamespace(pid=1111, poll=lambda: None)
    active = [SimpleNamespace(live=True)]
    snapshot = ResourceSnapshot(
        _host(0), _gpu(0), 100, 100, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), True, True, 0,
    )
    launches = []
    resume_order = []

    monkeypatch.setattr(
        scheduler_module, "terminate_process_tree",
        lambda process: resume_order.append(("terminate", process.pid)),
    )

    def launch(runtime, **kwargs):
        launches.append(kwargs)
        resume_order.append(("launch", runtime.pause_reason))
        runtime.process = SimpleNamespace(pid=1234, poll=lambda: None)
        runtime.pid = runtime.process.pid
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = kwargs["device"]

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    assert coordinator._resume(paused, active, snapshot, concurrency=1)
    assert resume_order == [("terminate", 1111), ("launch", "")]
    assert launches[0]["device"] == "gpu"
    assert launches[0]["launch_context"]["resumed_from_pause"] is True
    assert launches[0]["launch_context"]["checkpoint_path"] == str(
        (tmp_path / "result" / "estimator.checkpoint.pt").resolve()
    )
    assert paused in active


def test_zero_retries_is_uncapped_and_failure_requeues_to_tail(tmp_path, monkeypatch):
    from pathlib import Path
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="probe", seed=4,
        split_sha256="split", run_signature="suite-signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"), {}, ("python", "-c", "pass"))
    runtime = JobRuntime(spec, status=RunStatus.RUNNING, attempt=1, device_mode="cpu")
    runtime.process = SimpleNamespace(poll=lambda: 17)
    runtime.log_path = str(tmp_path / "job.log")
    Path(runtime.log_path).write_text("=== LAUNCH 1 ===\nordinary failure\n")
    runtime.log_handle = Path(runtime.log_path).open("a", encoding="utf-8")
    plan = SimpleNamespace(run_signature="suite-signature", runs=(spec,))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(max_retries=0), run_root=tmp_path / "run",
        results_dir=tmp_path / "result", repository_root=tmp_path,
    )
    monkeypatch.setattr(coordinator, "_classify_failure", lambda _runtime, code: f"nonzero_exit:{code}")
    snapshot = SimpleNamespace(gpu=None, as_dict=lambda: {})
    queue = deque(["already-queued"])
    coordinator._handle_exit(runtime, [runtime], [], queue, snapshot)
    assert list(queue) == ["already-queued", runtime]
    assert runtime.status == RunStatus.RETRYING
    assert runtime.retry_count == 1
    # A completed failed attempt keeps its end timestamp while retrying; the
    # source clears it only when the next launch starts.
    assert runtime.exit_code is None and runtime.ended_at is not None
    state = coordinator.state_store.load(runtime.job_id)
    assert state["exit_code"] is None and state["last_exit_code"] == 17
    assert state["ended_at"] == runtime.ended_at
    assert runtime.log_handle is None
    assert Path(runtime.log_path).read_text().endswith(f"=== RETRY {runtime.job_id} exit=17 ===\n")
    with coordinator.state_store.events_path.open(newline="", encoding="utf-8") as stream:
        retry_event = list(csv.DictReader(stream))[-1]
    assert retry_event["event"] == "retry" and retry_event["exit_code"] == "17"
    assert coordinator.counters.retries == 1
    assert coordinator._should_retry(coordinator.settings, runtime)


def test_controller_pass_orders_resume_then_reap_then_pressure_then_new_admission(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.state import JobRuntime, RunStatus

    def make_spec(route_name):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=route_name, seed=4,
            split_sha256="split", run_signature="suite-signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        route = ModelRoute("ADP-W-All")
        return RunSpec(identity, ref, "probe", route, "split.npz", str(tmp_path / route_name), {}, ("python", "-c", "pass"))

    class ExitNow:
        pid = 10
        def poll(self):
            return 0

    class LiveThenExit:
        pid = 11
        def __init__(self):
            self.calls = 0
        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 0

    active_runtime = JobRuntime(make_spec("already-active"), status=RunStatus.RUNNING, device_mode="gpu", process=ExitNow())
    paused_runtime = JobRuntime(make_spec("paused"), status=RunStatus.PAUSED, device_mode="gpu", pause_reason="host_ram_pressure")
    queued_runtime = JobRuntime(make_spec("queued"))
    specs = (active_runtime.spec, paused_runtime.spec, queued_runtime.spec)
    plan = SuitePlan("suite-signature", {"run_signature": "suite-signature", "jobs": []}, specs)
    settings = RunnerSettings(concurrency=4, post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1)
    coordinator = SuiteCoordinator(
        plan, settings, run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    events = []
    host = _host(8192)
    gpu = _gpu(8192)
    snapshot = ResourceSnapshot(host, gpu, 1, 1, 0, 0, 0, 0, 0, 0, frozenset(), frozenset(), False, False, 10)
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([active_runtime], [paused_runtime], deque([queued_runtime])))
    monkeypatch.setattr(coordinator, "_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *args: (events.append("sample") or snapshot, None, None))
    monkeypatch.setattr(coordinator, "_resume_ready", lambda *args: (events.append("resume_check") or True))
    original_resume = coordinator._resume
    def record_resume(*args, **kwargs):
        events.append("resume")
        return original_resume(*args, **kwargs)
    monkeypatch.setattr(coordinator, "_resume", record_resume)
    original_exit = coordinator._handle_exit
    def record_exit(*args, **kwargs):
        events.append("reap")
        return original_exit(*args, **kwargs)
    monkeypatch.setattr(coordinator, "_handle_exit", record_exit)
    original_pressure = coordinator._suspend_pressure_victim
    def record_pressure(*args, **kwargs):
        events.append("pressure")
        return original_pressure(*args, **kwargs)
    monkeypatch.setattr(coordinator, "_suspend_pressure_victim", record_pressure)
    def fake_launch(runtime, **kwargs):
        events.append("resume_launch" if kwargs.get("launch_context", {}).get("resumed_from_pause") else "queue_launch")
        runtime.process = LiveThenExit()
        runtime.pid = runtime.process.pid
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = kwargs["device"]
        return runtime.process
    monkeypatch.setattr(scheduler_module, "launch_job", fake_launch)
    monkeypatch.setattr(scheduler_module, "choose_device", lambda *args, **kwargs: "gpu")
    monkeypatch.setattr(coordinator, "_write_final_reports", lambda: {"completed": 3})
    coordinator.sleep = lambda _seconds: None

    coordinator.run()
    first_cycle = events[:events.index("sample", 1)]
    assert first_cycle.index("sample") < first_cycle.index("resume_check")
    assert first_cycle.index("resume") < first_cycle.index("reap")
    assert first_cycle.index("reap") < first_cycle.index("pressure")
    assert first_cycle.index("pressure") < first_cycle.index("queue_launch")
    assert first_cycle.count("queue_launch") == 1


def test_peer_oom_priority_uses_ledger_depth_width_job_id_tuple(tmp_path):
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.state import JobRuntime

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="route-z", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    route = ModelRoute("ADP-WD-All-Dcur", init_width=8, init_depth=3)
    spec = RunSpec(identity, ref, "probe", route, "split.npz", str(tmp_path), {"init_width": 1, "init_depth": 1}, ("python",))
    runtime = JobRuntime(spec)
    assert SuiteCoordinator._architecture_priority(runtime) == (3, 8, identity.run_id)

    fixed = ModelRoute("MLP-Fixed", fixed_grid=((4, 4, 4),))
    fixed_spec = RunSpec(identity, ref, "probe", fixed, "split.npz", str(tmp_path), {}, ("python",))
    assert SuiteCoordinator._architecture_priority(JobRuntime(fixed_spec)) == (3, 4, identity.run_id)


def test_pressure_victim_ties_use_depth_width_job_id_not_fold_sample(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    runtimes = []
    for job_id, depth, fold in (("shallow", 2, 99), ("deep", 4, 0)):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=fold,
            sample=fold, suite_repeat=1, model_route=job_id, seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        route = ModelRoute("ADP-W-All", init_width=4, init_depth=depth)
        spec = RunSpec(identity, ref, "probe", route, "split.npz", str(tmp_path), {}, ("python",))
        runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode="gpu", started_at=fold)
        runtime.pid = fold + 100
        runtime.process = SimpleNamespace(poll=lambda: None)
        runtimes.append(runtime)
    assert all(runtime.live and runtime.is_gpu for runtime in runtimes)

    monkeypatch.setattr(scheduler_module, "sample_child_memory", lambda *_args, **_kwargs: ProcessMemorySample(10, 20))
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    suspended = []
    selected = []
    monkeypatch.setattr(coordinator, "_pause", lambda runtime, *_args: selected.append(runtime) or True)
    snapshot = SimpleNamespace(
        gpu_pressure=True, host_pressure=False, gpu=SimpleNamespace(used_mib=100, free_mib=250),
        as_dict=lambda: {},
    )
    coordinator._suspend_pressure_victim(runtimes, suspended, snapshot)
    assert selected == [runtimes[1]]


def test_combined_host_gpu_pressure_uses_gpu_victim_rules_and_pauses_one(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    runtimes = []
    for name, device, depth in (
        ("cpu-high-rss", "cpu", 5),
        ("gpu-low-vram", "gpu", 2),
        ("gpu-high-vram", "gpu", 1),
    ):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=name, seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        route = ModelRoute("ADP-W-All", init_width=4, init_depth=depth)
        spec = RunSpec(identity, ref, "probe", route, "split.npz", str(tmp_path), {}, ("python",))
        runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode=device)
        runtime.pid = len(runtimes) + 10
        runtime.process = SimpleNamespace(pid=runtime.pid, poll=lambda: None)
        runtimes.append(runtime)

    samples = {
        runtimes[0].job_id: ProcessMemorySample(host_rss_mib=900, gpu_vram_mib=0),
        runtimes[1].job_id: ProcessMemorySample(host_rss_mib=100, gpu_vram_mib=50),
        runtimes[2].job_id: ProcessMemorySample(host_rss_mib=1, gpu_vram_mib=200),
    }
    monkeypatch.setattr(
        scheduler_module, "sample_child_memory",
        lambda pid, **_kwargs: samples[next(r.job_id for r in runtimes if r.pid == pid)],
    )
    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    selected = []
    monkeypatch.setattr(coordinator, "_pause", lambda runtime, _active, _suspended, reason, _snapshot: selected.append((runtime, reason)) or True)
    snapshot = SimpleNamespace(
        gpu_pressure=True, host_pressure=True,
        gpu=SimpleNamespace(used_mib=7900),
        host=SimpleNamespace(total_mib=8192, available_mib=1000),
        as_dict=lambda: {"gpu_pressure": True, "host_pressure": True},
    )

    coordinator._suspend_pressure_victim(runtimes, [], snapshot)

    assert selected == [(runtimes[2], "gpu_memory_pressure+host_ram_pressure")]
    assert coordinator.memory.values["gpu_peak_vram_mib"] == 7900
    assert coordinator.memory.values["cpu_peak_host_mib"] == 7192


@pytest.mark.parametrize("host_pressure,gpu_pressure", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("devices", [(), ("cpu",), ("gpu",), ("cpu", "gpu")])
def test_pressure_victim_policy_matrix_across_pressure_and_live_lanes(
    tmp_path, monkeypatch, host_pressure, gpu_pressure, devices,
):
    """Exercise the source rule across every pressure/lane-presence combination."""
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    runtimes = []
    samples = {}
    for index, device in enumerate(devices):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="matrix", split_repeat=0,
            fold=index, sample=index, suite_repeat=1, model_route=f"{device}-{index}",
            seed=4, split_sha256="split", run_signature="signature",
        )
        spec = RunSpec(
            identity, DatasetTaskRef("toy", "suite", "matrix"), "probe",
            ModelRoute("ADP-W-All", init_width=4, init_depth=index + 1),
            "split.npz", str(tmp_path), {}, ("python",),
        )
        runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode=device)
        runtime.pid = 100 + index
        runtime.process = SimpleNamespace(pid=runtime.pid, poll=lambda: None)
        runtimes.append(runtime)
        # Deliberately make CPU largest by host RSS and GPU largest by VRAM.
        samples[runtime.pid] = ProcessMemorySample(
            host_rss_mib=900 if device == "cpu" else 100,
            gpu_vram_mib=10 if device == "cpu" else 300,
        )

    monkeypatch.setattr(
        scheduler_module, "sample_child_memory",
        lambda pid, **_kwargs: samples[pid],
    )
    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    selected = []
    monkeypatch.setattr(
        coordinator, "_pause",
        lambda runtime, _active, _suspended, reason, _snapshot:
            selected.append((runtime, reason)) or True,
    )
    snapshot = SimpleNamespace(
        host_pressure=host_pressure, gpu_pressure=gpu_pressure,
        host=SimpleNamespace(total_mib=8192, available_mib=1000),
        gpu=SimpleNamespace(used_mib=7900),
        as_dict=lambda: {"host_pressure": host_pressure, "gpu_pressure": gpu_pressure},
    )

    coordinator._suspend_pressure_victim(runtimes, [], snapshot)

    should_pause = (host_pressure or gpu_pressure) and bool(devices) and (not gpu_pressure or "gpu" in devices)
    if not should_pause:
        assert selected == []
    else:
        assert len(selected) == 1
        runtime, reason = selected[0]
        if gpu_pressure:
            assert runtime.device_mode == "gpu"
            assert reason == ("gpu_memory_pressure+host_ram_pressure" if host_pressure else "gpu_memory_pressure")
        else:
            assert reason == "host_ram_pressure"
            expected_lane = "cpu" if "cpu" in devices else "gpu"
            assert runtime.device_mode == expected_lane
        if host_pressure:
            assert coordinator.memory.values["cpu_peak_host_mib"] == 7192
        if gpu_pressure:
            assert coordinator.memory.values["gpu_peak_vram_mib"] == 7900


@pytest.mark.parametrize("host_pressure,gpu_pressure", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("devices", [("cpu",), ("gpu",), ("cpu", "gpu")])
def test_full_controller_pass_applies_pressure_policy_across_live_lanes(
    tmp_path, monkeypatch, host_pressure, gpu_pressure, devices,
):
    """Verify the whole pass (including shutdown persistence) for each live lane mix."""
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample, ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    active = []
    samples = {}
    for index, device in enumerate(devices):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="pass-matrix", split_repeat=0,
            fold=index, sample=0, suite_repeat=1, model_route=f"{device}-{index}",
            seed=8, split_sha256="split", run_signature="signature",
        )
        spec = RunSpec(
            identity, DatasetTaskRef("toy", "suite", "pass-matrix"), "probe",
            ModelRoute("ADP-W-All", init_width=4, init_depth=index + 1),
            "split.npz", str(tmp_path / f"result-{device}-{index}"), {}, ("python",),
        )
        pid = 8100 + index
        runtime = JobRuntime(
            spec, status=RunStatus.RUNNING, attempt=1, device_mode=device, pid=pid,
            process=SimpleNamespace(pid=pid, poll=lambda: None),
        )
        active.append(runtime)
        samples[pid] = ProcessMemorySample(
            host_rss_mib=900 if device == "cpu" else 100,
            gpu_vram_mib=20 if device == "cpu" else 300,
        )

    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(r.spec for r in active))
    original_runtimes = tuple(active)
    settings = RunnerSettings(
        concurrency=4, run_hardness_screen=False, post_launch_sample_delay_seconds=0,
        pressure_poll_seconds=0.1,
    )
    provider = SimpleNamespace(sample_child=lambda pid, **_kwargs: samples[pid])
    coordinator = SuiteCoordinator(
        plan, settings, run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=provider,
    )
    host = _host(1000 if host_pressure else 4096)
    gpu = _gpu(50 if gpu_pressure else 4096)
    snapshot = ResourceSnapshot(
        host, gpu, len(active), sum(runtime.is_gpu for runtime in active),
        sum(not runtime.is_gpu for runtime in active), 0, 0, 0, 0, 0,
        frozenset(), frozenset(), host_pressure, gpu_pressure, 1,
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: (active, [], deque()))
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    coordinator.sleep = lambda _seconds: setattr(coordinator, "shutdown_signal", int(signal.SIGTERM))

    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    if coordinator.state_store.events_path.exists():
        with coordinator.state_store.events_path.open(encoding="utf-8", newline="") as stream:
            events = list(csv.DictReader(stream))
    else:
        events = []
    pauses = [row for row in events if row["event"] == "pause"]
    should_pause = (host_pressure or gpu_pressure) and (not gpu_pressure or "gpu" in devices)
    assert len(pauses) == int(should_pause)
    if should_pause:
        if gpu_pressure:
            expected = next(runtime for runtime in original_runtimes if runtime.device_mode == "gpu")
            expected_reason = "gpu_memory_pressure+host_ram_pressure" if host_pressure else "gpu_memory_pressure"
        else:
            expected = next(runtime for runtime in original_runtimes if runtime.device_mode == "cpu") if "cpu" in devices else original_runtimes[0]
            expected_reason = "host_ram_pressure"
        assert pauses[0]["job_id"] == expected.job_id
        assert pauses[0]["reason"] == expected_reason
        assert expected.status == RunStatus.PAUSED
    assert all(runtime.status in {RunStatus.PAUSED, RunStatus.QUEUED} for runtime in original_runtimes)


def test_successful_hard_pressure_pause_clears_transient_backoff_state(tmp_path, monkeypatch, caplog):
    import logging
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="pressure-backoff", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All", init_width=4, init_depth=2),
        "split.npz", str(tmp_path), {}, ("python",),
    )
    runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode="gpu")
    runtime.pid = 17
    runtime.process = SimpleNamespace(pid=17, poll=lambda: None)
    monkeypatch.setattr(scheduler_module, "sample_child_memory", lambda *_args, **_kwargs: ProcessMemorySample(10, 20))

    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    coordinator.admission.pressure_backoff_pending = True
    coordinator.admission.last_backoff_reason = "cuda_oom"
    coordinator.memory.values["last_backoff_ts"] = "prior-backoff"
    monkeypatch.setattr(coordinator, "_pause", lambda *_args: True)
    snapshot = SimpleNamespace(
        gpu_pressure=True, host_pressure=False, gpu=SimpleNamespace(used_mib=100, free_mib=250),
        host=SimpleNamespace(available_mib=300), active_rss_mib=444,
        as_dict=lambda: {"gpu_pressure": True},
    )

    active = [runtime]
    suspended = []
    def pause_and_move(victim, active_jobs, suspended_jobs, *_args):
        active_jobs.remove(victim)
        victim.status = RunStatus.PAUSED
        suspended_jobs.append(victim)
        return True
    monkeypatch.setattr(coordinator, "_pause", pause_and_move)
    with caplog.at_level(logging.WARNING, logger="simple_adp.runner"):
        coordinator._suspend_pressure_victim(active, suspended, snapshot)

    persisted = coordinator.batch_state_path.read_text(encoding="utf-8")
    state = __import__("json").loads(persisted)
    assert coordinator.admission.pressure_backoff_pending is False
    assert coordinator.admission.last_backoff_reason is None
    assert coordinator.memory.values["last_backoff_ts"] is None
    assert state["pressure_backoff_pending"] is False
    assert state["last_backoff_reason"] is None
    assert state["last_backoff_ts"] is None
    records = [record.getMessage() for record in caplog.records if "[PRESSURE]" in record.getMessage()]
    assert records == [
        "[PRESSURE] paused job=" + runtime.job_id
        + " reason=gpu_memory_pressure active=0 suspended=1 host_avail=300 host_rss=444 gpu_free=250 gpu_used=100"
    ]


def test_status_log_matches_source_fields_and_thirty_second_throttle(tmp_path, caplog):
    import logging

    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="status-signature", runs=()),
        RunnerSettings(run_hardness_screen=False), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
        wall_clock=lambda: 100.0,
    )
    coordinator.last_resource_snapshot = {
        "host": {"available_mib": 1234},
        "gpu": {"free_mib": 555, "used_mib": 445},
        "active_rss_mib": 77,
        "launch_capacity": 3,
    }
    coordinator.memory.values.update({
        "last_child_host_mib": 12,
        "last_child_gpu_mib": 34,
    })
    coordinator.admission.batch_scale = 0.5
    coordinator.admission.gpu_launches_enabled = False
    coordinator.admission.cpu_launches_enabled = True
    coordinator.admission.gpu_gate_closed = True
    coordinator.admission.host_gate_closed = False
    coordinator.admission.gpu_hold_until = 110.0
    coordinator.admission.cpu_hold_until = 99.0

    with caplog.at_level(logging.INFO, logger="simple_adp.runner"):
        coordinator._status([], [], deque(["queued-a", "queued-b"]), force=True)
        coordinator._status([], [], deque(), force=False)

    messages = [record.getMessage() for record in caplog.records if record.getMessage().startswith("[STATUS]")]
    assert messages == [
        "[STATUS] queued=2 active=0 suspended=0 completed=0 failed_attempts=0 terminal_failed=0 retried=0 "
        "batches_done=0 epochs_done=0 live_batches_done=0 live_epochs_done=0 telemetry_jobs=0 telemetry_dirs=0 "
        "batch_scale=0.50 gpu_launches=False cpu_launches=True gpu_gate_closed=True host_gate_closed=False "
        "gpu_hold=10.0s cpu_hold=0.0s host_avail=1234 host_rss=77 gpu_free=555 gpu_used=445 "
        "last_child_rss=12 last_child_vram=34 launch_cap=3"
    ]
def test_full_pass_reopens_external_host_gate_resumes_paused_job_and_reaps_exit(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="paused-cpu", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(tmp_path / "result"), {}, ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))

    class SyntheticResources:
        def sample_host(self):
            return _host(4096)

        def sample_gpu(self, _device_index):
            return None

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, _pid, **_kwargs):
            return ProcessMemorySample()

        def sample_external_host(self, _excluded):
            return ProcessMemorySample(host_rss_mib=400, pid_set=frozenset())

        def sample_external_gpu(self, _device_index, _excluded):
            return ProcessMemorySample()

    coordinator = SuiteCoordinator(
        plan, RunnerSettings(cpu_only=True, run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=SyntheticResources(),
        poll_sleep=lambda _seconds: None,
    )
    coordinator.memory.values.update({
        "external_host_mib": 1000,
        "external_host_pids": [90],
        "cpu_peak_host_mib": 0,
    })
    coordinator.admission.host_gate_closed = True
    coordinator.admission.cpu_launches_enabled = False
    paused = JobRuntime(
        spec, status=RunStatus.PAUSED, device_mode="cpu",
        pause_reason="host_ram_pressure",
    )
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([], [paused], deque()))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    class Exited:
        pid = 717

        def poll(self):
            return 0

    launch_contexts = []

    def launch(runtime, **kwargs):
        launch_contexts.append(kwargs["launch_context"])
        runtime.process = Exited()
        runtime.pid = runtime.process.pid
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = kwargs["device"]

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    summary = coordinator.run()

    assert coordinator.last_resource_snapshot["external_host_reopened"] is True
    assert coordinator.admission.host_gate_closed is False
    assert coordinator.admission.cpu_launches_enabled is True
    assert len(launch_contexts) == 1
    assert launch_contexts[0]["resumed_from_pause"] is True
    assert coordinator.state_store.events_path.read_text().count(",resume,") == 1
    assert summary["completed"] == 1


def test_full_pass_reopens_external_gpu_gate_resumes_on_same_lane_and_reaps_exit(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="paused-gpu", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(tmp_path / "result"), {}, ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))

    class SyntheticResources:
        def sample_host(self):
            return _host(4096)

        def sample_gpu(self, _device_index):
            return _gpu(600)

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, _pid, **_kwargs):
            return ProcessMemorySample()

        def sample_external_host(self, _excluded):
            return ProcessMemorySample()

        def sample_external_gpu(self, _device_index, _excluded):
            # A 600 MiB external drop plus disappearance of the prior PID
            # reopens the GPU gate; the 600 MiB free sample also meets resume.
            return ProcessMemorySample(gpu_vram_mib=400, pid_set=frozenset())

    coordinator = SuiteCoordinator(
        plan, RunnerSettings(run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=SyntheticResources(),
        poll_sleep=lambda _seconds: None,
    )
    coordinator.memory.values.update({
        "external_gpu_mib": 1000,
        "external_gpu_pids": [90],
        "gpu_peak_vram_mib": 0,
    })
    coordinator.admission.gpu_gate_closed = True
    coordinator.admission.gpu_gate_lock_job_id = "external-gpu-owner"
    coordinator.admission.gpu_launches_enabled = False
    paused = JobRuntime(
        spec, status=RunStatus.PAUSED, device_mode="gpu",
        pause_reason="gpu_memory_pressure",
    )
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([], [paused], deque()))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    class Exited:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 0

    launches = []

    def launch(runtime, *, device, launch_context, **_kwargs):
        launches.append((device, launch_context))
        runtime.process = Exited(818)
        runtime.pid = runtime.process.pid
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = device
        runtime.attempt += 1
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    summary = coordinator.run()

    assert coordinator.last_resource_snapshot["external_gpu_reopened"] is True
    assert coordinator.last_resource_snapshot["external_host_reopened"] is False
    assert coordinator.admission.gpu_gate_closed is False
    assert coordinator.admission.gpu_gate_lock_job_id is None
    assert coordinator.admission.gpu_launches_enabled is True
    assert launches[0][0] == "gpu"
    assert launches[0][1]["resumed_from_pause"] is True
    assert paused.pid == 818
    assert paused.status == RunStatus.COMPLETED
    assert summary["completed"] == 1
    assert coordinator.state_store.events_path.read_text().count(",resume,") == 1


def test_resume_requires_matching_lane_threshold_after_other_domain_reopens(tmp_path):
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="gpu-after-host-drop", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(tmp_path), {}, ("python",),
    )
    runtime = JobRuntime(
        spec, status=RunStatus.PAUSED, device_mode="gpu",
        pause_reason="host_ram_pressure",
    )
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="signature", runs=(spec,)), RunnerSettings(),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )

    below_gpu_resume = ResourceSnapshot(
        _host(1000), _gpu(499), 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 1,
    )
    at_gpu_resume = ResourceSnapshot(
        _host(1000), _gpu(500), 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 1,
    )

    # An external host drop can satisfy the stored host-pressure reason, but
    # the matching GPU lane still needs its unconditional VRAM resume level.
    assert not coordinator._resume_ready(runtime, below_gpu_resume, host_drop=500, gpu_drop=0)
    assert coordinator._resume_ready(runtime, at_gpu_resume, host_drop=500, gpu_drop=0)


def test_full_pass_external_drop_opens_gate_but_does_not_resume_below_lane_threshold(tmp_path, monkeypatch):
    import signal

    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="paused-gpu-below-resume", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"), {},
        ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))

    class SyntheticResources:
        def sample_host(self):
            return _host(4096)

        def sample_gpu(self, _device_index):
            return _gpu(499)

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, _pid, **_kwargs):
            return ProcessMemorySample()

        def sample_external_host(self, _excluded_pids):
            return ProcessMemorySample()

        def sample_external_gpu(self, _device_index, _excluded_pids):
            # This drop plus disappearance of the previously tracked PID opens
            # the GPU admission gate, but free VRAM is still below 500 MiB.
            return ProcessMemorySample(gpu_vram_mib=400, pid_set=frozenset())

    coordinator = SuiteCoordinator(
        plan, RunnerSettings(run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=SyntheticResources(),
    )
    coordinator.memory.values.update({"external_gpu_mib": 1000, "external_gpu_pids": [90]})
    coordinator.admission.gpu_gate_closed = True
    coordinator.admission.gpu_gate_lock_job_id = "external-owner"
    coordinator.admission.gpu_launches_enabled = False
    paused = JobRuntime(
        spec, status=RunStatus.PAUSED, device_mode="gpu",
        pause_reason="gpu_memory_pressure",
    )
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([], [paused], deque()))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    launches = []
    monkeypatch.setattr(scheduler_module, "launch_job", lambda *_args, **_kwargs: launches.append(True))

    def stop_after_one_pass(_seconds):
        coordinator.shutdown_signal = int(signal.SIGTERM)

    coordinator.sleep = stop_after_one_pass
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert coordinator.last_resource_snapshot["external_gpu_reopened"] is True
    assert coordinator.admission.gpu_gate_closed is False
    assert coordinator.admission.gpu_launches_enabled is True
    assert launches == []
    assert paused.status == RunStatus.PAUSED
    assert paused.pause_reason == "gpu_memory_pressure"


def test_full_pressure_pause_relaunches_on_same_gpu_with_new_pid(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.artifacts import estimator_checkpoint_path
    from simple_adp.runner.resources import GPUMemorySample, ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="gpu-pause-resume", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All", init_width=4, init_depth=1), "split.npz",
        str(tmp_path / "result"), {}, ("python", "-c", "pass"),
    )

    class ChangingResources:
        def __init__(self):
            self.pass_number = 0

        def sample_host(self):
            self.pass_number += 1
            return _host(500 if self.pass_number == 1 else 4096)

        def sample_gpu(self, _device_index):
            free = 50 if self.pass_number == 1 else 4096
            total = 8192
            used = total - free
            return GPUMemorySample(total, used, free, used / total * 100)

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, _pid, **_kwargs):
            return ProcessMemorySample(host_rss_mib=200, gpu_vram_mib=100)

        def sample_external_host(self, _excluded):
            return ProcessMemorySample()

        def sample_external_gpu(self, _device_index, _excluded):
            return ProcessMemorySample()

    runtime = JobRuntime(spec, status=RunStatus.RUNNING, device_mode="gpu", pid=101)
    runtime.process = SimpleNamespace(pid=101, poll=lambda: None)
    coordinator = SuiteCoordinator(
        SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,)),
        RunnerSettings(run_hardness_screen=False, pressure_poll_seconds=0.1),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=ChangingResources(),
        poll_sleep=lambda _seconds: None,
    )
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([runtime], [], deque()))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    terminated = []
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda process: terminated.append(process.pid))
    launches = []

    class Completed:
        pid = 202

        def poll(self):
            return 0

    def launch(job, **kwargs):
        launches.append((job.device_mode, kwargs["device"], kwargs["launch_context"]))
        job.process = Completed()
        job.pid = job.process.pid
        job.status = RunStatus.RUNNING
        job.device_mode = kwargs["device"]

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    summary = coordinator.run()

    assert terminated == [101]
    assert runtime.pause_count == runtime.gpu_pause_count == runtime.host_pause_count == 1
    assert len(launches) == 1
    assert launches[0][1] == "gpu"
    assert launches[0][2]["resumed_from_pause"] is True
    assert launches[0][2]["pause_reason"] == "gpu_memory_pressure+host_ram_pressure"
    assert launches[0][2]["resume_device"] == "gpu"
    assert launches[0][2]["checkpoint_path"] == str(
        estimator_checkpoint_path(runtime.spec.result_dir)
    )
    job_dir = coordinator.state_store.job_dir(runtime)
    assert launches[0][2]["checkpoint_last"] == str(job_dir / "checkpoint_last.pt")
    assert launches[0][2]["checkpoint_best"] == str(job_dir / "checkpoint_best.pt")
    assert launches[0][2]["checkpoint_best_batch"] == str(job_dir / "checkpoint_best_batch.pt")
    assert runtime.pid == 202
    assert summary["completed"] == 1
    assert coordinator.state_store.load(runtime.job_id)["status"] == RunStatus.COMPLETED.value


def test_normal_drain_zeros_log_counts_before_final_status_then_writes_reports(tmp_path, monkeypatch):
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []}, (),
    )
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(run_hardness_screen=False), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
        poll_sleep=lambda _seconds: None,
    )
    coordinator.logging_counts = LaneCounts(total=3, gpu=2, cpu=1)
    events = []

    def record_status(*_args, force=False, **_kwargs):
        if force:
            events.append(("final_status", coordinator.logging_counts.gpu, coordinator.logging_counts.cpu))

    def record_reports():
        events.append(("reports", coordinator.logging_counts.gpu, coordinator.logging_counts.cpu))
        return {"completed": 0}

    monkeypatch.setattr(coordinator, "_status", record_status)
    monkeypatch.setattr(coordinator, "_write_final_reports", record_reports)

    assert coordinator.run() == {"completed": 0}
    assert events == [("final_status", 0, 0), ("reports", 0, 0)]


def test_final_reports_cover_all_runtime_summary_fields_and_task_rows(tmp_path):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="probe", seed=4,
        split_sha256="split", run_signature="suite-signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    result_dir = tmp_path / "results" / "task-1"
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(result_dir), {}, ("python", "-c", "pass"),
    )
    plan = SuitePlan("suite-signature", {"run_signature": "suite-signature", "jobs": [spec.as_dict()]}, (spec,))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(run_hardness_screen=False), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    runtime = __import__("simple_adp.runner.state", fromlist=["JobRuntime", "RunStatus"])
    job = runtime.JobRuntime(spec, status=runtime.RunStatus.COMPLETED, attempt=2,
                             retry_count=1, exit_code=0, last_exit_code=1)
    coordinator.state_store.save(job)
    progress = result_dir / "telemetry" / "progress_state.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({"history_rows": 7, "final_epoch": 3}), encoding="utf-8")
    coordinator.counters.completed = 1
    coordinator.counters.failed_attempts = 1
    coordinator.counters.retries = 1
    coordinator.counters.recovered_unknown_exit = 2

    summary = coordinator._write_final_reports()
    expected_fields = {
        "run_root", "results_dir", "run_signature", "manifest_jobs", "completed",
        "terminal_failures", "failed_attempts", "retries", "recovered_unknown_exit",
        "timestamp", "resource_snapshot", "hardness_screening", "batches_done",
        "epochs_done", "live_batches_done", "live_epochs_done", "telemetry_jobs",
        "telemetry_dirs",
    }
    assert set(summary) == expected_fields
    assert summary["manifest_jobs"] == summary["completed"] == 1
    assert summary["failed_attempts"] == summary["retries"] == 1
    assert summary["recovered_unknown_exit"] == 2
    assert (summary["batches_done"], summary["epochs_done"]) == (7, 3)
    assert (summary["live_batches_done"], summary["live_epochs_done"]) == (0, 0)
    assert (summary["telemetry_jobs"], summary["telemetry_dirs"]) == (1, 1)

    run_root = tmp_path / "run"
    for name in ("final_report.json", "suite_summary.json"):
        assert json.loads((run_root / name).read_text(encoding="utf-8")) == summary
    final_markdown = (run_root / "final_report.md").read_text(encoding="utf-8")
    suite_markdown = (run_root / "suite_summary.md").read_text(encoding="utf-8")
    for fragment in (
        "- Total jobs: 1", "- Completed: 1", "- Terminal failed: 0",
        "- Failed attempts: 1", "- Retries: 1", "- Batches done: 7",
        "- Epochs done: 3", "- Live-job batches done: 0", "- Live-job epochs done: 0",
    ):
        assert fragment in final_markdown
        assert fragment in suite_markdown
    assert "- Timestamp:" in final_markdown
    assert "- Timestamp:" not in suite_markdown

    with (run_root / "task_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        ["job_id", "status", "attempt", "retry_count", "exit_code", "last_exit_code", "result_dir"],
        [identity.run_id, "completed", "2", "1", "0", "1", str(result_dir)],
    ]


@pytest.mark.parametrize(
    ("device", "expected_gpu_gate", "expected_host_gate", "expected_gpu_lock", "expected_host_lock"),
    [
        ("cpu", True, False, "other-gpu-job", None),
        ("gpu", False, False, None, None),
    ],
)
def test_successful_completion_releases_device_gates_and_reenables_source_lanes(
    tmp_path, device, expected_gpu_gate, expected_host_gate, expected_gpu_lock, expected_host_lock,
):
    from simple_adp.runner.admission import AdmissionState
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route=f"{device}-job", seed=4,
        split_sha256="split", run_signature="suite-signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(tmp_path / "result"), {}, ("python", "-c", "pass"),
    )
    plan = SuitePlan("suite-signature", {"run_signature": "suite-signature", "jobs": []}, (spec,))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(run_hardness_screen=False), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    coordinator.admission = AdmissionState(
        gpu_gate_closed=True, host_gate_closed=True,
        gpu_gate_lock_job_id="other-gpu-job", host_gate_lock_job_id="other-cpu-job",
        gpu_launches_enabled=False, cpu_launches_enabled=False,
        batch_scale=0.5, pressure_backoff_pending=True, last_backoff_reason="cuda_oom",
    )
    coordinator.memory.values["last_backoff_ts"] = "prior_timestamp"
    runtime = JobRuntime(spec, device_mode=device, exit_code=0)

    coordinator._finalize(runtime, success=True)

    assert coordinator.admission.gpu_gate_closed is expected_gpu_gate
    assert coordinator.admission.host_gate_closed is expected_host_gate
    assert coordinator.admission.gpu_gate_lock_job_id == expected_gpu_lock
    assert coordinator.admission.host_gate_lock_job_id == expected_host_lock
    # The source branch only enables GPU launches after GPU completion. CPU
    # completion leaves that flag unchanged even if the GPU gate stays closed.
    assert coordinator.admission.gpu_launches_enabled is (device == "gpu")
    assert coordinator.admission.cpu_launches_enabled is True
    assert coordinator.admission.batch_scale == 0.5
    assert coordinator.admission.pressure_backoff_pending is False
    assert coordinator.admission.last_backoff_reason is None
    assert coordinator.memory.values["last_backoff_ts"] is None
    persisted_backoff = json.loads(coordinator.batch_state_path.read_text(encoding="utf-8"))
    assert persisted_backoff["last_backoff_ts"] is None
    assert runtime.status == RunStatus.COMPLETED


def test_full_pass_cpu_completion_keeps_gpu_gate_closed_and_uses_cpu_fallback(tmp_path, monkeypatch):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    specs = []
    for route in ("finished-cpu", "queued-next"):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=route, seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        specs.append(RunSpec(
            identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
            str(tmp_path / route), {}, ("python", "-c", "pass"),
        ))
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(specs))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(concurrency=2, post_launch_sample_delay_seconds=0,
                             run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, poll_sleep=lambda _seconds: None,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    class Process:
        def __init__(self, pid, code=None):
            self.pid = pid
            self.code = code

        def poll(self):
            return self.code

    finished = JobRuntime(
        specs[0], status=RunStatus.RUNNING, device_mode="cpu", process=Process(41, 0), pid=41,
    )
    queued = JobRuntime(specs[1])
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([finished], [], deque([queued])))
    snapshot = ResourceSnapshot(
        host=_host(4096), gpu=_gpu(4096), active_count=1, active_gpu_count=0,
        active_cpu_count=1, active_rss_mib=0, latest_child_rss_mib=0,
        latest_child_vram_mib=0, external_host_mib=0, external_gpu_mib=0,
        external_host_pids=frozenset(), external_gpu_pids=frozenset(),
        host_pressure=False, gpu_pressure=False, launch_capacity=2,
    )

    def sample(*_args):
        coordinator.admission.gpu_gate_closed = True
        coordinator.admission.gpu_gate_lock_job_id = "another-live-job"
        coordinator.admission.gpu_launches_enabled = False
        return snapshot, None, None

    monkeypatch.setattr(coordinator, "_sample_snapshot", sample)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launched = []

    def launch(runtime, *, device, **_kwargs):
        runtime.process = Process(99)
        runtime.pid = 99
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = device
        launched.append(device)
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)

    def stop_after_launch(_seconds):
        if coordinator.counters.launched:
            coordinator.shutdown_signal = int(signal.SIGTERM)

    coordinator.sleep = stop_after_launch
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert coordinator.counters.completed == 1
    assert launched == ["cpu"]
    assert coordinator.admission.gpu_gate_closed is True
    assert coordinator.admission.gpu_gate_lock_job_id == "another-live-job"
    assert coordinator.admission.gpu_launches_enabled is False
    assert coordinator.admission.cpu_launches_enabled is True


@pytest.mark.parametrize(
    ("gpu_pressure", "expected_pause_reason"),
    [(True, None), (False, "host_ram_pressure")],
)
def test_full_pass_combined_pressure_prefers_gpu_victim_set_even_without_gpu_jobs(
    tmp_path, monkeypatch, gpu_pressure, expected_pause_reason,
):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample, ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    def make_spec(name):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=name, seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        return RunSpec(
            identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
            str(tmp_path / name), {}, ("python", "-c", "pass"),
        )

    active_spec, queued_spec = make_spec("cpu-active"), make_spec("queued-route")
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (active_spec, queued_spec))

    class LiveProcess:
        pid = 701

        def poll(self):
            return None

    class SyntheticResources:
        def sample_child(self, _pid, **_kwargs):
            return ProcessMemorySample(host_rss_mib=700, gpu_vram_mib=0)

    active = JobRuntime(
        active_spec, status=RunStatus.RUNNING, device_mode="cpu", pid=701,
        process=LiveProcess(), started_at=1,
    )
    queued = JobRuntime(queued_spec)
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(concurrency=2, run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=SyntheticResources(),
    )
    snapshot = ResourceSnapshot(
        host=_host(500), gpu=_gpu(50), active_count=1, active_gpu_count=0,
        active_cpu_count=1, active_rss_mib=700, latest_child_rss_mib=700,
        latest_child_vram_mib=0, external_host_mib=0, external_gpu_mib=0,
        external_host_pids=frozenset(), external_gpu_pids=frozenset(),
        host_pressure=True, gpu_pressure=gpu_pressure, launch_capacity=1,
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([active], [], deque([queued])))
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    monkeypatch.setattr(
        scheduler_module, "launch_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pressure must block admission")),
    )
    pause_attempts = []
    original_pause = coordinator._pause

    def record_pause(runtime, active_jobs, suspended_jobs, reason, current_snapshot):
        pause_attempts.append((runtime.job_id, reason))
        return original_pause(runtime, active_jobs, suspended_jobs, reason, current_snapshot)

    monkeypatch.setattr(coordinator, "_pause", record_pause)

    def stop_after_pass(_seconds):
        coordinator.shutdown_signal = int(signal.SIGTERM)

    coordinator.sleep = stop_after_pass
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    if gpu_pressure:
        # Combined pressure selects GPU victim rules. The sole CPU process is
        # ineligible, so the controller leaves it running during this pass.
        assert pause_attempts == []
        assert active.status == RunStatus.QUEUED  # shutdown requeues live work
        assert coordinator.counters.launched == 0
    else:
        assert pause_attempts == [(active.job_id, "host_ram_pressure")]
        assert active.status == RunStatus.PAUSED
        assert active.pause_reason == expected_pause_reason
        assert coordinator.counters.launched == 0


def test_posix_watchdog_terminates_descendants_then_root_then_kills_survivors(tmp_path, monkeypatch):
    import simple_adp.runner.watchdog as watchdog_module
    events = []

    class Process:
        def __init__(self, pid):
            self.pid = pid
        def terminate(self): events.append(("terminate", self.pid))
        def kill(self): events.append(("kill", self.pid))
        def is_running(self): return True
        def children(self, recursive=True):
            events.append(("children", self.pid, recursive))
            return [Process(12), Process(11)]

    monkeypatch.setattr(watchdog_module, "master_alive", lambda _pid: False)
    monkeypatch.setattr(watchdog_module.psutil, "Process", Process)
    monkeypatch.setattr(watchdog_module.time, "sleep", lambda seconds: events.append(("sleep", seconds)))
    watchdog_module.monitor(101, 202, tmp_path / "watchdog.log", initial_delay=0, grace_seconds=15)
    assert events == [
        ("sleep", 0), ("sleep", 3.0), ("children", 202, True),
        ("terminate", 11), ("terminate", 12), ("terminate", 202),
        ("sleep", 15),
        ("children", 202, True),
        ("kill", 11), ("kill", 12), ("kill", 202),
    ]


def test_windows_watchdog_waits_then_kills_descendants_in_descending_pid_order(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import simple_adp.runner.watchdog as watchdog_module

    events = []
    alive_states = iter((True, True, False))

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def terminate(self):
            events.append(("terminate", self.pid))

        def kill(self):
            events.append(("kill", self.pid))

        def is_running(self):
            return True

        def children(self, recursive=True):
            events.append(("children", self.pid, recursive))
            return [Process(11), Process(12)]

    monkeypatch.setattr(watchdog_module, "master_alive", lambda _pid: next(alive_states))
    monkeypatch.setattr(watchdog_module.psutil, "Process", Process)
    monkeypatch.setattr(watchdog_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(watchdog_module.time, "sleep", lambda seconds: events.append(("sleep", seconds)))

    watchdog_module.monitor(101, 202, tmp_path / "watchdog.log")

    assert events == [
        ("sleep", 2.0), ("sleep", 3.0), ("sleep", 3.0), ("sleep", 3.0),
        ("terminate", 202), ("sleep", 15.0),
        ("children", 202, True), ("kill", 12), ("kill", 11), ("kill", 202),
    ]


def test_process_tree_termination_returns_for_already_exited_popen(monkeypatch):
    import subprocess
    import sys
    import simple_adp.runner.processes as process_module

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    assert child.wait(timeout=5) == 0
    monkeypatch.setattr(process_module.psutil, "Process", lambda _pid: (_ for _ in ()).throw(AssertionError("must not inspect a completed root")))
    process_module.terminate_process_tree(child)


def test_process_tree_termination_reverses_descendants_and_kills_survivors_twice(monkeypatch):
    import simple_adp.runner.processes as process_module

    events = []

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.child_processes = []

        def children(self, *, recursive):
            events.append(("children", self.pid, recursive))
            return self.child_processes

        def terminate(self):
            events.append(("terminate", self.pid))

        def kill(self):
            events.append(("kill", self.pid))

    root = Process(10)
    root.child_processes = [Process(11), Process(12)]
    first_survivors = [Process(12), root]
    second_survivors = [root]
    wait_results = iter((first_survivors, second_survivors))

    def wait_procs(processes, *, timeout):
        events.append(("wait", [process.pid for process in processes], timeout))
        return [], next(wait_results)

    monkeypatch.setattr(process_module.psutil, "Process", lambda _pid: root)
    monkeypatch.setattr(process_module.psutil, "wait_procs", wait_procs)

    process_module.terminate_process_tree(10, timeout_seconds=0)

    assert events == [
        ("children", 10, True),
        ("terminate", 12), ("terminate", 11), ("terminate", 10),
        ("wait", [10, 11, 12], 0.1),
        ("kill", 12), ("kill", 10),
        ("wait", [12, 10], 0.1),
        ("kill", 10),
    ]


def test_process_tree_termination_falls_back_to_popen_when_psutil_cannot_open_root(monkeypatch):
    import subprocess
    import sys
    import simple_adp.runner.processes as process_module

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    calls = []
    original_terminate = child.terminate

    def record_terminate():
        calls.append("popen_terminate")
        original_terminate()

    child.terminate = record_terminate
    monkeypatch.setattr(
        process_module.psutil, "Process",
        lambda pid: (_ for _ in ()).throw(process_module.psutil.NoSuchProcess(pid)),
    )
    monkeypatch.setattr(
        process_module.psutil, "wait_procs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback must return without waiting")),
    )

    try:
        process_module.terminate_process_tree(child)
        assert calls == ["popen_terminate"]
    finally:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def test_process_tree_termination_waits_for_surviving_root_without_popen_kill(monkeypatch):
    import subprocess
    import sys
    import simple_adp.runner.processes as process_module

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    events = []

    class Root:
        pid = child.pid

        def children(self, *, recursive):
            events.append(("children", recursive))
            return []

        def terminate(self):
            events.append(("terminate",))

        def kill(self):
            events.append(("psutil_kill",))

    root = Root()
    waits = iter(([root], [root]))

    def wait_procs(processes, *, timeout):
        events.append(("wait_procs", [item.pid for item in processes], timeout))
        return [], next(waits)

    original_kill = child.kill
    child.kill = lambda: events.append(("popen_kill",))
    child.wait = lambda *, timeout=None: events.append(("popen_wait", timeout)) or 0
    monkeypatch.setattr(process_module.psutil, "Process", lambda _pid: root)
    monkeypatch.setattr(process_module.psutil, "wait_procs", wait_procs)

    try:
        process_module.terminate_process_tree(child, timeout_seconds=0.25)
        assert events == [
            ("children", True), ("terminate",),
            ("wait_procs", [child.pid], 0.25), ("psutil_kill",),
            ("wait_procs", [child.pid], 0.25), ("psutil_kill",),
            ("popen_wait", 0.25),
        ]
    finally:
        child.kill = original_kill
        original_kill()
        subprocess.Popen.wait(child, timeout=5)


def test_process_tree_termination_propagates_descendant_enumeration_failure(monkeypatch):
    import simple_adp.runner.processes as process_module

    events = []

    class Root:
        pid = 321

        def children(self, *, recursive):
            events.append(("children", recursive))
            raise process_module.psutil.AccessDenied(self.pid)

        def terminate(self):
            events.append(("terminate",))

    monkeypatch.setattr(process_module.psutil, "Process", lambda _pid: Root())
    monkeypatch.setattr(
        process_module.psutil, "wait_procs",
        lambda *_args, **_kwargs: pytest.fail("enumeration failure must propagate before waiting"),
    )

    with pytest.raises(process_module.psutil.AccessDenied):
        process_module.terminate_process_tree(321)

    assert events == [("children", True)]


def test_source_run_root_copy_order_is_named_first_and_never_overwrites(tmp_path, monkeypatch):
    import simple_adp.runner.state as state_module

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    (destination).mkdir()
    for name in ("manifest.json", "job_manifest.json", "run_metadata.json", "extra.json"):
        (source / name).write_text(f"source:{name}")
    (source / "nested" / "state.json").write_text("nested state")
    (destination / "run_metadata.json").write_text("keep existing")
    copied_sources = []
    original_copy2 = state_module.shutil.copy2

    def record_copy(src, dst):
        copied_sources.append(Path(src).relative_to(source).as_posix())
        return original_copy2(src, dst)

    from pathlib import Path
    monkeypatch.setattr(state_module.shutil, "copy2", record_copy)
    counts = state_module.seed_run_root_from_source(source, destination)
    assert counts == {"copied": 4, "skipped": 1}
    assert copied_sources[:2] == ["manifest.json", "job_manifest.json"]
    assert set(copied_sources[2:]) == {"extra.json", "nested/state.json"}
    assert (destination / "run_metadata.json").read_text() == "keep existing"
    assert (destination / "manifest.json").read_text() == "source:manifest.json"
    assert (destination / "nested" / "state.json").read_text() == "nested state"


def _host(available=4096):
    return HostMemorySample(8192, available, 50, 0, 0, 0)


def _gpu(free=4096):
    total = 8192
    used = total - free
    return GPUMemorySample(total, used, free, used / total * 100)


def test_parent_cpu_mask_routes_to_cpu_even_when_gpu_is_available(monkeypatch):
    settings = RunnerSettings()
    state = AdmissionState()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("TABULAR_BACKEND", raising=False)
    monkeypatch.delenv("TABULAR_CPU_ONLY", raising=False)
    monkeypatch.delenv("TRAINING_CONTROL_CPU_ONLY", raising=False)
    monkeypatch.delenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", raising=False)
    monkeypatch.delenv("TRAINING_CONTROL_BACKEND", raising=False)
    monkeypatch.delenv("CPU_ONLY", raising=False)
    monkeypatch.delenv("OPF_ADP_DISABLE_GPU_ACCELERATORS", raising=False)
    assert parent_cpu_masked({"CUDA_VISIBLE_DEVICES": "  -1 "})
    assert parent_cpu_masked({"TABULAR_BACKEND": "CPU"})
    assert parent_cpu_masked({"TRAINING_CONTROL_CPU_ONLY": "yes"})
    for key, value in (
        ("TRAINING_CONTROL_BACKEND", " CPU "),
        ("TABULAR_BACKEND", "CPU"),
        ("CPU_ONLY", " YeS "),
        ("TABULAR_CPU_ONLY", " YeS "),
        ("TRAINING_CONTROL_CPU_ONLY", " YeS "),
        ("OPF_ADP_DISABLE_GPU_ACCELERATORS", " YeS "),
        ("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", " YeS "),
    ):
        assert parent_cpu_masked({key: value})
    assert not parent_cpu_masked({"CUDA_VISIBLE_DEVICES": "0"})
    assert choose_device(
        settings,
        state,
        host=_host(),
        gpu=_gpu(),
        counts=LaneCounts(0, 0, 0),
        now=1,
    ) == "gpu"
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert resolve_torch_device() == "cpu"


def test_launcher_preserves_parent_cpu_only_mask(monkeypatch):
    import simple_adp.runner.runtime_tuning as tuning

    monkeypatch.setattr(tuning, "_apply_affinity", lambda _cpus: None)
    result = launcher_child_env(
        {"CUDA_VISIBLE_DEVICES": "", "TABULAR_BACKEND": "", "OMP_NUM_THREADS": "8"},
        job_key="parent-cpu", launch_budget=2, cpu_only=False,
    )
    assert result["CUDA_VISIBLE_DEVICES"] == ""
    assert result["TABULAR_BACKEND"] == "cpu"
    assert result["SIMPLE_ADP_DISABLE_GPU_ACCELERATORS"] == "1"
    for mask in (
        {"TRAINING_CONTROL_BACKEND": "CPU"},
        {"CPU_ONLY": "yes"},
        {"OPF_ADP_DISABLE_GPU_ACCELERATORS": "1"},
    ):
        child = launcher_child_env({"CUDA_VISIBLE_DEVICES": "0", **mask}, job_key="legacy-mask", launch_budget=2)
        assert child["CUDA_VISIBLE_DEVICES"] == ""
        assert child["TABULAR_BACKEND"] == "cpu"


def test_admission_capacity_strict_pressure_gates_and_resume_predicates(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    settings = RunnerSettings(launch_host_reserve_mib=1024, launch_gpu_reserve_mib=1024)
    assert resolve_launch_capacity(settings, concurrency=20, host_available_mib=3072, gpu=_gpu(2048), active_count=0) == 2
    assert resolve_launch_capacity(settings, concurrency=20, host_available_mib=100, gpu=_gpu(100), active_count=0) == 1
    assert resolve_launch_capacity(settings, concurrency=20, host_available_mib=8192, gpu=_gpu(), active_count=3) == 4
    assert resolve_launch_capacity(settings, concurrency=20, host_available_mib=4096, gpu=None, active_count=0) == 4
    capped = RunnerSettings(
        launch_host_reserve_mib=1024, launch_gpu_reserve_mib=1024,
        max_active_jobs=1,
    )
    assert resolve_launch_capacity(capped, concurrency=20, host_available_mib=8192, gpu=_gpu(), active_count=0) == 1
    unbounded = RunnerSettings()
    assert resolve_launch_capacity(
        unbounded, concurrency=10**9, host_available_mib=0, gpu=None, active_count=99,
    ) == 10**9
    host_pressure, gpu_pressure = hard_pressure(
        RunnerSettings(), _host(1024), GPUMemorySample(8192, 8092, 100, 98.8)
    )
    assert (host_pressure, gpu_pressure) == (False, False)  # exact thresholds use strict '<'


    state = AdmissionState()
    update_soft_gates(
        state, settings=RunnerSettings(), host_pressure=False, gpu_pressure=False,
        gpu_sample_present=True, latest_gpu_job="g1", latest_host_job="g1",
        latest_child_gpu_mib=401, latest_child_host_mib=1000,
        host_available_mib=1200, gpu_free_mib=500,
    )
    assert state.gpu_gate_closed and state.host_gate_closed
    assert state.gpu_gate_lock_job_id == state.host_gate_lock_job_id == "g1"
    assert not choose_device(
        RunnerSettings(), state, host=_host(4096), gpu=_gpu(),
        counts=LaneCounts(0, 0, 0), now=100,
    )
    defaults = RunnerSettings()
    assert pressure_pause_reopened(
        pause_reason="host_ram_pressure+gpu_memory_pressure", device="gpu",
        host_available_mib=1280, gpu_free_mib=500, host_used_mib=6912, gpu_used_mib=7692,
        settings=defaults, host_peak_used_mib=0, gpu_peak_used_mib=0,
        external_host_drop_mib=0, external_gpu_drop_mib=0,
    )
    assert not pressure_pause_reopened(
        pause_reason="host_ram_pressure+gpu_memory_pressure", device="gpu",
        host_available_mib=1280, gpu_free_mib=499, host_used_mib=6912, gpu_used_mib=7693,
        settings=defaults, host_peak_used_mib=0, gpu_peak_used_mib=0,
        external_host_drop_mib=0, external_gpu_drop_mib=0,
    )


def test_reserve_capacity_preserves_python_float_floor_division_boundaries():
    settings = RunnerSettings(launch_host_reserve_mib=0.1, launch_gpu_reserve_mib=0.1)
    # The copied runner uses `int(value // reserve)`: binary float floor
    # division returns 9 for 1.0 // 0.1, while floor(1.0 / 0.1) returns 10.
    assert resolve_launch_capacity(
        settings, concurrency=20, host_available_mib=1.0,
        gpu=_gpu(1.0), active_count=0,
    ) == 9


def test_soft_gate_reassignment_only_occurs_when_gate_transitions_closed():
    # The source sets lock IDs and lane flags only on an open -> closed
    # transition. An already-closed gate can retain its old diagnostic owner
    # and enabled flag after completion has reset lane flags.
    state = AdmissionState(
        gpu_gate_closed=True,
        host_gate_closed=True,
        gpu_gate_lock_job_id="prior-gpu-owner",
        host_gate_lock_job_id="prior-host-owner",
        gpu_launches_enabled=True,
        cpu_launches_enabled=True,
    )
    update_soft_gates(
        state,
        settings=RunnerSettings(),
        host_pressure=True,
        gpu_pressure=True,
        gpu_sample_present=True,
        latest_gpu_job="new-gpu-runtime",
        latest_host_job="new-host-runtime",
        latest_child_gpu_mib=800,
        latest_child_host_mib=4000,
        host_available_mib=100,
        gpu_free_mib=50,
    )
    assert state.gpu_gate_closed and state.host_gate_closed
    assert state.gpu_gate_lock_job_id == "prior-gpu-owner"
    assert state.host_gate_lock_job_id == "prior-host-owner"
    assert state.gpu_launches_enabled and state.cpu_launches_enabled


@pytest.mark.parametrize(
    ("pause_reason", "device", "ready_domain", "ready_inputs"),
    [
        ("host_ram_pressure", "cpu", "host_threshold", {}),
        ("host_ram_pressure", "cpu", "host_peak_drop", {"host_peak_used_mib": 1000, "host_used_mib": 500}),
        ("host_ram_pressure", "cpu", "host_external_drop", {"external_host_drop_mib": 500}),
        ("gpu_memory_pressure", "gpu", "gpu_threshold", {}),
        ("gpu_memory_pressure", "gpu", "gpu_peak_drop", {"gpu_peak_used_mib": 1000, "gpu_used_mib": 500}),
        ("gpu_memory_pressure", "gpu", "gpu_external_drop", {"external_gpu_drop_mib": 500}),
        # A combined cause requires both domains to reopen, even when either
        # domain alone would satisfy its own resume predicate.
        ("host_ram_pressure+gpu_memory_pressure", "gpu", "host_only", {"external_host_drop_mib": 500}),
        ("host_ram_pressure+gpu_memory_pressure", "gpu", "gpu_only", {"external_gpu_drop_mib": 500}),
    ],
)
def test_pressure_resume_predicate_covers_each_domain_and_combined_causes(
    pause_reason, device, ready_domain, ready_inputs,
):
    settings = RunnerSettings()
    values = {
        "host_available_mib": 100,
        "gpu_free_mib": 50,
        "host_used_mib": 7000,
        "gpu_used_mib": 8000,
        "host_peak_used_mib": 0,
        "gpu_peak_used_mib": 0,
        "external_host_drop_mib": 0,
        "external_gpu_drop_mib": 0,
    }
    if ready_domain == "host_threshold":
        values["host_available_mib"] = settings.host_ram_resume_avail_mib
    elif ready_domain == "gpu_threshold":
        values["gpu_free_mib"] = settings.gpu_resume_avail_mib
    elif ready_domain == "host_only":
        pass
    elif ready_domain == "gpu_only":
        pass
    values.update(ready_inputs)

    opened = pressure_pause_reopened(
        pause_reason=pause_reason,
        device=device,
        settings=settings,
        **values,
    )
    assert opened is (ready_domain not in {"host_only", "gpu_only"})


@pytest.mark.parametrize("scheduler", ["pressure_aware", "gpu_first", "fixed"])
def test_gpu_admission_also_requires_host_resume_and_labels_match_source(monkeypatch, scheduler):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for key in (
        "TABULAR_BACKEND", "TABULAR_CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
        "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "TRAINING_CONTROL_BACKEND", "CPU_ONLY",
        "OPF_ADP_DISABLE_GPU_ACCELERATORS",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = RunnerSettings(scheduler=scheduler)
    state = AdmissionState()
    gpu = _gpu()
    empty = LaneCounts(0, 0, 0)

    assert choose_device(
        settings, state, host=_host(1280), gpu=gpu, counts=empty, now=1,
    ) == "gpu"  # resume threshold equality is admitted
    assert choose_device(
        settings, state, host=_host(1279), gpu=gpu, counts=empty, now=1,
    ) is None  # GPU admission requires the host resume threshold too

    # With both lanes available, each configured label chooses GPU; all fall
    # back to CPU when the GPU sensor is absent.
    assert choose_device(
        settings, state, host=_host(4096), gpu=gpu, counts=empty, now=1,
    ) == "gpu"
    assert choose_device(
        settings, state, host=_host(4096), gpu=None, counts=empty, now=1,
    ) == "cpu"

    # The source's pressure_aware label also requires CPU-lane readiness. A
    # live CPU post-launch hold can therefore suppress a GPU-ready launch.
    held_cpu_state = AdmissionState(cpu_hold_until=2)
    assert choose_device(
        settings, held_cpu_state, host=_host(4096), gpu=gpu,
        counts=empty, now=1,
    ) == (None if scheduler == "pressure_aware" else "gpu")


def test_soft_gate_equality_external_pid_reopen_and_lane_cap_hold_edges(monkeypatch):
    from simple_adp.runner.resources import external_drop_reopened

    state = AdmissionState()
    update_soft_gates(
        state, settings=RunnerSettings(), host_pressure=False, gpu_pressure=False,
        gpu_sample_present=True, latest_gpu_job="g1", latest_host_job="g1",
        latest_child_gpu_mib=400, latest_child_host_mib=700,
        host_available_mib=1200, gpu_free_mib=500,
    )
    # Buffered equality is open; the source gates use strict greater-than.
    assert not state.gpu_gate_closed
    assert not state.host_gate_closed
    assert not external_drop_reopened(
        previous_mib=1000, current_mib=400,
        previous_pids={10}, current_pids={10}, threshold_mib=500,
    )
    assert external_drop_reopened(
        previous_mib=1000, current_mib=400,
        previous_pids={10}, current_pids=set(), threshold_mib=500,
    )
    # The external reopen gate accepts the exact drop threshold, but still
    # requires a positive previous sample and disappearance of a tracked PID.
    assert external_drop_reopened(
        previous_mib=1000, current_mib=500,
        previous_pids={10}, current_pids=set(), threshold_mib=500,
    )
    assert not external_drop_reopened(
        previous_mib=1000, current_mib=500.01,
        previous_pids={10}, current_pids=set(), threshold_mib=500,
    )
    assert not external_drop_reopened(
        previous_mib=0, current_mib=0,
        previous_pids={10}, current_pids=set(), threshold_mib=500,
    )
    assert not external_drop_reopened(
        previous_mib=1000, current_mib=500,
        previous_pids={10}, current_pids={10}, threshold_mib=500,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    host = _host(4096)
    gpu = _gpu(8192)
    cpu_capped = RunnerSettings(max_active_jobs=1, cpu_only=True)
    assert choose_device(
        cpu_capped, AdmissionState(), host=host, gpu=gpu,
        counts=LaneCounts(1, 0, 1), now=10,
    ) == "cpu"
    gpu_capped = RunnerSettings(max_active_gpu_jobs=1)
    assert choose_device(
        gpu_capped, AdmissionState(), host=host, gpu=gpu,
        counts=LaneCounts(1, 1, 0), now=10,
    ) == "cpu"
    gpu_hold = AdmissionState(gpu_hold_until=11)
    assert choose_device(
        RunnerSettings(), gpu_hold, host=host, gpu=gpu,
        counts=LaneCounts(0, 0, 0), now=10,
    ) == "cpu"


def test_soft_gate_updater_exhaustive_pressure_lane_sensor_and_buffer_product():
    from itertools import product

    checked = 0
    for (
        gpu_live, host_live, gpu_sensor, gpu_pressure, host_pressure,
        gpu_buffer_relation, host_buffer_relation,
    ) in product(
        (False, True), (False, True), (False, True), (False, True), (False, True),
        (-1, 0, 1), (-1, 0, 1),
    ):
        settings = RunnerSettings(
            child_gpu_gate_buffer_mib=100,
            child_host_gate_buffer_mib=500,
        )
        state = AdmissionState()
        latest_gpu_job = "gpu-job" if gpu_live else None
        latest_host_job = "latest-job" if host_live else None
        child_gpu = 200.0
        gpu_free = child_gpu + settings.child_gpu_gate_buffer_mib + gpu_buffer_relation
        child_host = 700.0
        host_available = child_host + settings.child_host_gate_buffer_mib + host_buffer_relation

        update_soft_gates(
            state, settings=settings,
            host_pressure=host_pressure, gpu_pressure=gpu_pressure,
            gpu_sample_present=gpu_sensor,
            latest_gpu_job=latest_gpu_job, latest_host_job=latest_host_job,
            latest_child_gpu_mib=child_gpu, latest_child_host_mib=child_host,
            host_available_mib=host_available,
            gpu_free_mib=gpu_free if gpu_sensor else None,
        )

        expected_gpu_closed = gpu_live and gpu_sensor and (
            gpu_pressure or child_gpu + settings.child_gpu_gate_buffer_mib > gpu_free
        )
        expected_host_closed = host_live and (
            host_pressure or child_host + settings.child_host_gate_buffer_mib > host_available
        )
        assert state.gpu_gate_closed is expected_gpu_closed
        assert state.host_gate_closed is expected_host_closed
        assert state.gpu_gate_lock_job_id == ("gpu-job" if expected_gpu_closed else None)
        assert state.host_gate_lock_job_id == ("latest-job" if expected_host_closed else None)
        assert state.gpu_launches_enabled is (not expected_gpu_closed)
        assert state.cpu_launches_enabled is (not expected_host_closed)
        checked += 1
    assert checked == 2**5 * 3**2

@pytest.mark.parametrize(
    ("case", "host_available", "gpu_free", "cpu_only", "gpu_enabled", "cpu_enabled",
     "gpu_gate", "host_gate", "gpu_cap", "active_cap", "gpu_hold", "cpu_hold",
     "counts", "expected"),
    [
        ("default", 4096, 4096, False, True, True, False, False, 0, 0, 0, 0, LaneCounts(0, 0, 0), "gpu"),
        ("cpu_only", 4096, 4096, True, True, True, False, False, 0, 0, 0, 0, LaneCounts(0, 0, 0), "cpu"),
        ("gpu_launches_disabled", 4096, 4096, False, False, True, False, False, 0, 0, 0, 0, LaneCounts(0, 0, 0), "cpu"),
        ("gpu_gate_closed", 4096, 4096, False, True, True, True, False, 0, 0, 0, 0, LaneCounts(0, 0, 0), "cpu"),
        ("host_gate_closed", 4096, 4096, False, True, True, False, True, 0, 0, 0, 0, LaneCounts(0, 0, 0), None),
        ("gpu_sensor_missing", 4096, None, False, True, True, False, False, 0, 0, 0, 0, LaneCounts(0, 0, 0), "cpu"),
        ("gpu_below_resume", 4096, 499, False, True, True, False, False, 0, 0, 0, 0, LaneCounts(0, 0, 0), "cpu"),
        ("gpu_cap_full", 4096, 4096, False, True, True, False, False, 1, 0, 0, 0, LaneCounts(1, 1, 0), "cpu"),
        # Total admission capacity is enforced by the controller before it
        # calls choose_device; this selector only blocks the CPU fallback.
        ("all_active_cap_full", 4096, 4096, False, True, True, False, False, 0, 1, 0, 0, LaneCounts(1, 1, 0), "gpu"),
        ("gpu_hold_expired_later", 4096, 4096, False, True, True, False, False, 0, 0, 11, 0, LaneCounts(0, 0, 0), "cpu"),
        ("cpu_launches_disabled_fallback", 4096, 499, False, True, False, False, False, 0, 0, 0, 0, LaneCounts(0, 0, 0), None),
        ("host_below_resume", 1279, 4096, False, True, True, False, False, 0, 0, 0, 0, LaneCounts(0, 0, 0), None),
        ("cpu_hold_active_gpu_available", 4096, 4096, False, True, True, False, False, 0, 0, 0, 11, LaneCounts(0, 0, 0), "gpu"),
        ("gpu_hold_active_cpu_available", 4096, 499, False, True, True, False, False, 0, 0, 11, 0, LaneCounts(0, 0, 0), "cpu"),
    ],
)
def test_device_selection_branch_matrix(
    monkeypatch, case, host_available, gpu_free, cpu_only, gpu_enabled, cpu_enabled,
    gpu_gate, host_gate, gpu_cap, active_cap, gpu_hold, cpu_hold, counts, expected,
):
    import simple_adp.runner.admission as admission

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(admission, "parent_cpu_masked", lambda: False)
    settings = RunnerSettings(
        cpu_only=cpu_only,
        max_active_gpu_jobs=gpu_cap,
        max_active_jobs=active_cap,
    )
    state = AdmissionState(
        gpu_launches_enabled=gpu_enabled,
        cpu_launches_enabled=cpu_enabled,
        gpu_gate_closed=gpu_gate,
        host_gate_closed=host_gate,
        gpu_hold_until=gpu_hold,
        cpu_hold_until=cpu_hold,
    )
    gpu = _gpu(gpu_free) if gpu_free is not None else None
    assert choose_device(
        settings, state, host=_host(host_available), gpu=gpu, counts=counts, now=10,
    ) == expected, case


def test_device_selection_exhaustive_admission_boolean_product(monkeypatch):
    """Cross every independent lane, gate, cap, sensor, mask, and hold switch."""
    from itertools import product
    import simple_adp.runner.admission as admission

    host_ready_values = (False, True)
    gpu_sensor_values = (False, True)
    gpu_ready_values = (False, True)
    binary_values = (False, True)
    checked = 0
    for (
        cpu_only, parent_masked, host_ready, gpu_sensor, gpu_ready,
        gpu_enabled, cpu_enabled, gpu_gate, host_gate, gpu_cap_full,
        total_cap_full, gpu_hold_ready, cpu_hold_ready,
    ) in product(
        binary_values, binary_values, host_ready_values, gpu_sensor_values,
        gpu_ready_values, binary_values, binary_values, binary_values,
        binary_values, binary_values, binary_values, binary_values,
        binary_values,
    ):
        monkeypatch.setattr(admission, "parent_cpu_masked", lambda: parent_masked)
        settings = RunnerSettings(
            cpu_only=cpu_only,
            max_active_gpu_jobs=1 if gpu_cap_full else 0,
            max_active_jobs=1 if total_cap_full else 0,
        )
        state = AdmissionState(
            gpu_launches_enabled=gpu_enabled,
            cpu_launches_enabled=cpu_enabled,
            gpu_gate_closed=gpu_gate,
            host_gate_closed=host_gate,
            gpu_hold_until=0 if gpu_hold_ready else 2,
            cpu_hold_until=0 if cpu_hold_ready else 2,
        )
        counts = LaneCounts(
            total=int(total_cap_full or gpu_cap_full),
            gpu=int(gpu_cap_full),
            cpu=int(total_cap_full and not gpu_cap_full),
        )
        host = _host(settings.host_ram_resume_avail_mib if host_ready else settings.host_ram_resume_avail_mib - 1)
        gpu = (
            _gpu(settings.gpu_resume_avail_mib if gpu_ready else settings.gpu_resume_avail_mib - 1)
            if gpu_sensor else None
        )

        gpu_intended = (
            not cpu_only and gpu_enabled and not gpu_gate and not host_gate
            and host_ready and gpu_sensor and gpu_ready and not gpu_cap_full
            and gpu_hold_ready
        )
        host_intended = (
            cpu_enabled and not host_gate and host_ready and not total_cap_full
            and cpu_hold_ready
        )
        if cpu_only or parent_masked:
            # The source selector's forced-CPU branch returns before checking
            # the total cap; the outer controller budget enforces that cap.
            expected = "cpu" if (cpu_enabled and not host_gate and host_ready and cpu_hold_ready) else None
        elif gpu_intended:
            expected = "gpu"
        elif host_intended:
            expected = "cpu"
        else:
            expected = None

        actual = choose_device(
            settings, state, host=host, gpu=gpu, counts=counts, now=1,
        )
        assert actual == expected, {
            "cpu_only": cpu_only, "parent_masked": parent_masked,
            "host_ready": host_ready, "gpu_sensor": gpu_sensor,
            "gpu_ready": gpu_ready, "gpu_enabled": gpu_enabled,
            "cpu_enabled": cpu_enabled, "gpu_gate": gpu_gate,
            "host_gate": host_gate, "gpu_cap_full": gpu_cap_full,
            "total_cap_full": total_cap_full, "gpu_hold_ready": gpu_hold_ready,
            "cpu_hold_ready": cpu_hold_ready,
        }
        checked += 1
    assert checked == 2**13


def test_mixed_device_latest_child_sampling_retains_source_order_quirk(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.state import JobRuntime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(scheduler_module, "process_tree_pids", lambda pid: frozenset({pid}))
    monkeypatch.setattr(scheduler_module, "sample_child_memory", lambda pid, **kwargs: {
        11: ProcessMemorySample(100, 900, frozenset({11})),
        22: ProcessMemorySample(200, 0, frozenset({22})),
    }[pid])
    monkeypatch.setattr(scheduler_module, "sample_external_host_memory", lambda _pids: ProcessMemorySample())
    monkeypatch.setattr(scheduler_module, "sample_external_gpu_memory", lambda *_args: ProcessMemorySample())

    def runtime(job_id, pid, device, started_at):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=job_id, seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        route = ModelRoute("ADP-W-All", init_width=4, init_depth=1)
        spec = RunSpec(identity, ref, "probe", route, "split.npz", str(tmp_path), {}, ("python",))
        value = JobRuntime(spec, device_mode=device, started_at=started_at)
        value.pid = pid
        value.process = SimpleNamespace(poll=lambda: None)
        return value

    gpu_runtime = runtime("gpu-job", 11, "gpu", 1.0)
    cpu_runtime = runtime("cpu-job", 22, "cpu", 2.0)
    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(child_gpu_gate_buffer_mib=100, child_host_gate_buffer_mib=5000),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
        host_sampler=lambda: _host(4096), gpu_sampler=lambda _index: _gpu(500),
    )
    snapshot, latest_all, latest_gpu = coordinator._sample_snapshot([gpu_runtime, cpu_runtime], [], 10)
    assert latest_all is cpu_runtime and latest_gpu is gpu_runtime
    # The source samples the general latest runtime first and uses that sample
    # for the GPU gate even though the separately selected GPU runtime differs.
    assert snapshot.latest_child_vram_mib == 0
    assert coordinator.memory.values["last_child_job_id"] == cpu_runtime.job_id
    assert coordinator.last_resource_snapshot["latest_child_job_id"] == cpu_runtime.job_id
    persisted_memory = json.loads(coordinator.batch_state_path.read_text(encoding="utf-8"))
    assert persisted_memory["last_child_job_id"] == cpu_runtime.job_id
    assert not coordinator.admission.gpu_gate_closed
    assert coordinator.admission.host_gate_closed
    # status/log context captures this sample before gate mutation; the live
    # admission object reflects the gate decision made from the same sample.
    assert coordinator.last_resource_snapshot["host_gate_closed"] is False
    record = logging.getLogger("simple-adp-resource-context-test").makeRecord(
        "simple-adp-resource-context-test", logging.INFO, __file__, 1, "sample", (), None,
    )
    assert record.host_total_mib == 8192
    assert record.host_available_mib == 4096
    assert record.cpu_processes == 1
    assert record.gpu_processes == 1


@pytest.mark.parametrize(("message", "expected"), [
    ("CUDA out of memory", "cuda_oom"),
    ("CUDNN_STATUS_ALLOC_FAILED", "cuda_oom"),
    ("CUBLAS_STATUS_ALLOC_FAILED", "cuda_oom"),
    ("CUBLASCreate(handle) failed", "cuda_oom"),
    ("out of memory in CUDA allocator", "cuda_oom"),
    ("cannot allocate memory", "memory_oom"),
    ("MemoryError: failed", "memory_oom"),
    ("process killed by OOM", "memory_oom"),
    ("ordinary nonzero exit", "nonzero_exit:9"),
])
def test_failure_marker_classification_matches_ledger(tmp_path, message, expected):
    from types import SimpleNamespace
    from simple_adp.runner.scheduler import SuiteCoordinator

    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    log = tmp_path / "job.log"
    log.write_text(f"=== LAUNCH 1 ===\n{message}\n")
    assert coordinator._classify_failure(SimpleNamespace(log_path=str(log)), 9) == expected


@pytest.mark.parametrize(
    ("old_attempt", "current_attempt", "expected"),
    [
        ("CUDA out of memory", "ordinary nonzero exit", "nonzero_exit:9"),
        ("ordinary nonzero exit", "CUDA out of memory", "cuda_oom"),
        ("exact-resume failure: old checkpoint", "ordinary nonzero exit", "nonzero_exit:9"),
    ],
)
def test_failure_classification_uses_only_the_latest_attempt(
    tmp_path, old_attempt, current_attempt, expected,
):
    from types import SimpleNamespace
    from simple_adp.runner.scheduler import SuiteCoordinator

    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    log = tmp_path / "retries.log"
    log.write_text(
        f"=== LAUNCH first ===\n{old_attempt}\n"
        f"=== LAUNCH second ===\n{current_attempt}\n",
        encoding="utf-8",
    )

    assert coordinator._classify_failure(SimpleNamespace(log_path=str(log)), 9) == expected


def test_exact_resume_marker_is_scanned_beyond_oom_tail_window(tmp_path):
    from types import SimpleNamespace
    from simple_adp.runner.scheduler import SuiteCoordinator

    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    log = tmp_path / "job.log"
    log.write_text("=== LAUNCH 1 ===\nexact-resume failure: absent RNG\n" + "x" * 5000)
    assert coordinator._classify_failure(SimpleNamespace(log_path=str(log)), 9) == "inexact_checkpoint_resume"


def test_corrupt_estimator_checkpoint_error_is_terminal_in_scheduler(tmp_path):
    from types import SimpleNamespace
    from simple_adp.checkpoint import CheckpointStore

    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"truncated checkpoint")
    with pytest.raises(ValueError, match="exact-resume failure") as failure:
        CheckpointStore(checkpoint).load(signature="expected")

    log = tmp_path / "attempt.log"
    log.write_text(f"=== LAUNCH 1 ===\n{failure.value}\n", encoding="utf-8")
    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    assert coordinator._classify_failure(SimpleNamespace(log_path=str(log)), 1) == "inexact_checkpoint_resume"


def test_concurrency_control_file_is_reread_and_zero_is_unbounded(tmp_path):
    control = tmp_path / "concurrency.txt"
    control.write_text("3\n")
    settings = RunnerSettings(concurrency=1, concurrency_file=str(control))
    assert settings.resolve_concurrency() == 3
    control.write_text("")
    assert settings.resolve_concurrency() == 1
    control.write_text("0\n")
    assert settings.resolve_concurrency() == 10**9
    control.write_text("-1")
    with pytest.raises(ValueError, match="non-negative integer"):
        settings.resolve_concurrency()
    control.write_text("not-an-integer")
    with pytest.raises(ValueError, match="non-negative integer"):
        settings.resolve_concurrency()
    missing = RunnerSettings(concurrency=1, concurrency_file=str(tmp_path / "missing.txt"))
    with pytest.raises(ValueError, match="cannot read concurrency file"):
        missing.resolve_concurrency()


def test_controller_rereads_concurrency_file_between_scheduling_passes(tmp_path, monkeypatch):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    control = tmp_path / "concurrency.txt"
    control.write_text("1\n", encoding="utf-8")
    specs = []
    for index in range(3):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=index,
            sample=0, suite_repeat=1, model_route=f"dynamic-{index}", seed=4,
            split_sha256="split", run_signature="signature",
        )
        specs.append(RunSpec(
            identity, DatasetTaskRef("toy", "suite", "1"), "probe",
            ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / f"result-{index}"),
            {}, ("python", "-c", "pass"),
        ))
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(specs))
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=1, concurrency_file=str(control), cpu_only=True,
            run_hardness_screen=False, post_launch_sample_delay_seconds=0,
            pressure_poll_seconds=0.1,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(
        _host(8192), None, 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 3,
    )
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launches = []

    def launch(runtime, *, device, **_kwargs):
        runtime.attempt += 1
        runtime.process = SimpleNamespace(pid=600 + len(launches), poll=lambda: None)
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        launches.append(runtime.job_id)
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)

    def advance(_seconds):
        if len(launches) == 1:
            control.write_text("2\n", encoding="utf-8")
        elif len(launches) == 2:
            coordinator.shutdown_signal = int(signal.SIGTERM)

    coordinator.sleep = advance
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert len(launches) == 2  # the updated limit admits a second live child


def test_queue_fill_rereads_concurrency_after_launch_failure(tmp_path, monkeypatch):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    control = tmp_path / "queue-concurrency.txt"
    control.write_text("2\n", encoding="utf-8")

    def make_spec(name, fold):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id=name, split_repeat=0,
            fold=fold, sample=0, suite_repeat=1, model_route=name, seed=4,
            split_sha256="split", run_signature="signature",
        )
        return RunSpec(
            identity, DatasetTaskRef("toy", "suite", name), "probe",
            ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / name), {},
            ("python", "-c", "pass"),
        )

    active = JobRuntime(make_spec("existing", 0), status=RunStatus.RUNNING, device_mode="cpu")
    active.pid = 500
    active.process = SimpleNamespace(pid=500, poll=lambda: None)
    first = JobRuntime(make_spec("first-fails", 1))
    second = JobRuntime(make_spec("second-waits", 2))
    queue = deque([first, second])
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []},
        (active.spec, first.spec, second.spec),
    )
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=2, concurrency_file=str(control), cpu_only=True,
            max_retries=1, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, wall_clock=lambda: 100.0,
    )
    snapshot = ResourceSnapshot(
        _host(8192), None, 1, 0, 1, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 2,
    )
    monkeypatch.setattr(
        coordinator, "_build_runtimes",
        lambda: ([active], [], queue),
    )
    monkeypatch.setattr(
        coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None),
    )
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launch_attempts = []

    def launch(runtime, *, launch_budget, **_kwargs):
        runtime.attempt += 1
        launch_attempts.append((runtime.job_id, launch_budget))
        if runtime is first:
            control.write_text("1\n", encoding="utf-8")
            raise OSError("controlled launch failure")
        raise AssertionError("lowered concurrency should prevent another queue launch")

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: setattr(
        coordinator, "shutdown_signal", int(signal.SIGTERM),
    )

    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert launch_attempts == [(first.job_id, 2)]
    assert first.status == RunStatus.RETRYING and first.retry_count == 1
    assert list(queue) == [second, first]
    assert coordinator.state_store.load(active.job_id)["status"] == "queued"


def test_lowering_live_concurrency_does_not_terminate_active_children(tmp_path, monkeypatch):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import RunStatus

    control = tmp_path / "concurrency-down.txt"
    control.write_text("2\n", encoding="utf-8")
    specs = []
    for index in range(3):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=index,
            sample=0, suite_repeat=1, model_route=f"lowering-{index}", seed=4,
            split_sha256="split", run_signature="signature",
        )
        specs.append(RunSpec(
            identity, DatasetTaskRef("toy", "suite", "1"), "probe",
            ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / f"lowered-result-{index}"),
            {}, ("python", "-c", "pass"),
        ))
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(specs))
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=2, concurrency_file=str(control), cpu_only=True,
            run_hardness_screen=False, post_launch_sample_delay_seconds=0,
            pressure_poll_seconds=0.1,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(
        _host(8192), None, 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 3,
    )
    observed_active = []
    lowered = False

    def sample(active, _suspended, _concurrency):
        statuses = tuple(runtime.status for runtime in active)
        observed_active.append(statuses)
        if lowered and len(active) == 2:
            coordinator.shutdown_signal = int(signal.SIGTERM)
        return snapshot, None, None

    monkeypatch.setattr(coordinator, "_sample_snapshot", sample)
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launches = []

    def launch(runtime, *, device, **_kwargs):
        runtime.attempt += 1
        runtime.process = SimpleNamespace(pid=700 + len(launches), poll=lambda: None)
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        launches.append(runtime.job_id)
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    def advance(_seconds):
        nonlocal lowered
        if len(launches) == 2 and not lowered:
            control.write_text("1\n", encoding="utf-8")
            lowered = True

    coordinator.sleep = advance
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert len(launches) == 2
    assert (RunStatus.RUNNING, RunStatus.RUNNING) in observed_active


@pytest.mark.parametrize("gpu_gate_closed", [False, True])
@pytest.mark.parametrize("host_gate_closed", [False, True])
@pytest.mark.parametrize("gpu_hold_active", [False, True])
@pytest.mark.parametrize("cpu_hold_active", [False, True])
def test_full_pass_combines_both_gates_and_lane_holds(
    tmp_path, monkeypatch, gpu_gate_closed, host_gate_closed,
    gpu_hold_active, cpu_hold_active,
):
    """Exercise the full queue-admission path across all gate/hold states."""
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import RunStatus

    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="gate-hold-matrix",
        split_repeat=0, fold=0, sample=0, suite_repeat=1,
        model_route="ADP-W-All", seed=19, split_sha256="split",
        run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "gate-hold-matrix"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"), {},
        ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=2, cpu_only=False, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
        wall_clock=lambda: 100.0,
    )
    snapshot = ResourceSnapshot(
        _host(4096), _gpu(4096), 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 2,
    )

    def sample(_active, _suspended, _concurrency):
        coordinator.admission.gpu_gate_closed = gpu_gate_closed
        coordinator.admission.host_gate_closed = host_gate_closed
        coordinator.admission.gpu_hold_until = 101.0 if gpu_hold_active else 0.0
        coordinator.admission.cpu_hold_until = 101.0 if cpu_hold_active else 0.0
        return snapshot, None, None

    monkeypatch.setattr(coordinator, "_sample_snapshot", sample)
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launches = []

    def launch(runtime, *, device, **_kwargs):
        runtime.attempt += 1
        runtime.process = SimpleNamespace(pid=901, poll=lambda: None)
        runtime.pid = 901
        runtime.device_mode = device
        runtime.started_at = 100.0
        runtime.status = RunStatus.RUNNING
        launches.append(device)
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)

    def stop_after_pass(_seconds):
        coordinator.shutdown_signal = int(signal.SIGTERM)

    coordinator.sleep = stop_after_pass
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()
    assert stopped.value.code == 130

    expected = (
        "gpu" if not gpu_gate_closed and not host_gate_closed and not gpu_hold_active
        else "cpu" if not host_gate_closed and not cpu_hold_active
        else None
    )
    assert launches == ([] if expected is None else [expected])


@pytest.mark.parametrize("peer_device", ["cpu", "gpu"])
@pytest.mark.parametrize("host_pressure,gpu_pressure", [
    (False, False), (True, False), (False, True), (True, True),
])
@pytest.mark.parametrize("gpu_gate_closed,host_gate_closed", [
    (False, False), (True, False), (False, True), (True, True),
])
@pytest.mark.parametrize("gpu_hold_active,cpu_hold_active", [
    (False, False), (True, False), (False, True), (True, True),
])
@pytest.mark.parametrize("max_active_jobs", [0, 1, 2])
@pytest.mark.parametrize("max_active_gpu_jobs", [0, 1, 2])
def test_full_pass_pressure_gate_hold_and_live_cap_cross_product(
    tmp_path, monkeypatch, peer_device, host_pressure, gpu_pressure,
    gpu_gate_closed, host_gate_closed, gpu_hold_active, cpu_hold_active,
    max_active_jobs, max_active_gpu_jobs,
):
    """Cross live-lane pressure, both gates/holds, and active-cap admission."""
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.admission import resolve_launch_capacity
    from simple_adp.runner.resources import ProcessMemorySample, ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    settings = RunnerSettings(
        concurrency=4,
        max_active_jobs=max_active_jobs,
        max_active_gpu_jobs=max_active_gpu_jobs,
        run_hardness_screen=False,
        post_launch_sample_delay_seconds=0,
        pressure_poll_seconds=0.1,
    )

    def make_runtime(name, device, depth):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id=name, split_repeat=0,
            fold=depth, sample=depth, suite_repeat=1, model_route=name,
            seed=19, split_sha256="split", run_signature="signature",
        )
        spec = RunSpec(
            identity, DatasetTaskRef("toy", "suite", name), "probe",
            ModelRoute("ADP-W-All", init_depth=depth, init_width=2),
            "split.npz", str(tmp_path / name), {}, ("python", "-c", "pass"),
        )
        return JobRuntime(spec, status=RunStatus.RUNNING, device_mode=device)

    peer = make_runtime("live-peer", peer_device, 2)
    peer.pid = 1200
    peer.process = SimpleNamespace(pid=peer.pid, poll=lambda: None)
    peer.started_at = 99.0
    waiting = make_runtime("waiting-route", "cpu", 3)
    waiting.status = RunStatus.QUEUED
    waiting.device_mode = None
    waiting.process = None
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []},
        (peer.spec, waiting.spec),
    )
    coordinator = SuiteCoordinator(
        plan, settings,
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
        wall_clock=lambda: 100.0,
        resource_provider=SimpleNamespace(
            sample_child=lambda _pid, **_kwargs: ProcessMemorySample(
                host_rss_mib=700 if peer_device == "cpu" else 100,
                gpu_vram_mib=700 if peer_device == "gpu" else 0,
            ),
        ),
    )
    host = _host(1023 if host_pressure else 4096)
    gpu = _gpu(99 if gpu_pressure else 4096)
    counts = coordinator._counts([peer])
    capacity = resolve_launch_capacity(
        settings, concurrency=settings.resolve_concurrency(),
        host_available_mib=host.available_mib, gpu=gpu,
        active_count=counts.total,
    )
    snapshot = ResourceSnapshot(
        host, gpu, counts.total, counts.gpu, counts.cpu,
        700 if peer_device == "cpu" else 100,
        700 if peer_device == "cpu" else 100,
        700 if peer_device == "gpu" else 0,
        0, 0, frozenset(), frozenset(),
        host_pressure, gpu_pressure, capacity,
    )

    def sample(_active, _suspended, _concurrency):
        coordinator.admission.gpu_gate_closed = gpu_gate_closed
        coordinator.admission.host_gate_closed = host_gate_closed
        coordinator.admission.gpu_hold_until = 101.0 if gpu_hold_active else 0.0
        coordinator.admission.cpu_hold_until = 101.0 if cpu_hold_active else 0.0
        return snapshot, peer, peer if peer_device == "gpu" else None

    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([peer], [], deque([waiting])))
    monkeypatch.setattr(coordinator, "_sample_snapshot", sample)
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launches = []

    def launch(runtime, *, device, **_kwargs):
        launches.append(device)
        runtime.attempt += 1
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        runtime.pid = 1300
        runtime.process = SimpleNamespace(pid=1300, poll=lambda: None)
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: setattr(
        coordinator, "shutdown_signal", int(signal.SIGTERM),
    )

    with pytest.raises(SystemExit) as stopped:
        coordinator.run()
    assert stopped.value.code == 130

    # The controller pauses at most one eligible victim before admission.
    peer_paused = (
        host_pressure and not gpu_pressure
        or gpu_pressure and peer_device == "gpu"
    )
    remaining_total = 0 if peer_paused else 1
    remaining_gpu = int(not peer_paused and peer_device == "gpu")
    total_cap_full = max_active_jobs > 0 and remaining_total >= max_active_jobs
    gpu_cap_full = max_active_gpu_jobs > 0 and remaining_gpu >= max_active_gpu_jobs
    gpu_ready = (
        not host_pressure and not gpu_pressure
        and not gpu_gate_closed and not host_gate_closed
        and not gpu_hold_active and not gpu_cap_full
    )
    cpu_ready = (
        not host_pressure and not host_gate_closed
        and not cpu_hold_active and not total_cap_full
    )
    expected_device = (
        None if total_cap_full
        else "gpu" if gpu_ready
        else "cpu" if cpu_ready
        else None
    )
    assert launches == ([] if expected_device is None else [expected_device])
    assert peer.status == (RunStatus.PAUSED if peer_paused else RunStatus.QUEUED)
    # Signal shutdown persists a successfully started route as queued for recovery.
    assert waiting.status == RunStatus.QUEUED
    assert waiting.attempt == (1 if expected_device is not None else 0)


@pytest.mark.parametrize("max_active_jobs", [0, 1, 2])
@pytest.mark.parametrize("max_active_gpu_jobs", [0, 1, 2])
def test_full_pass_combines_live_active_caps_with_gpu_fallback(
    tmp_path, monkeypatch, max_active_jobs, max_active_gpu_jobs,
):
    """A live GPU child plus one queued route exercises both active caps."""
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import CallbackResourceProvider, ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    identities = [
        RunIdentity(
            provider="toy", suite="suite", task_id="active-cap-matrix",
            split_repeat=0, fold=index, sample=0, suite_repeat=1,
            model_route=f"route-{index}", seed=43, split_sha256="split",
            run_signature="signature",
        )
        for index in range(2)
    ]
    refs = [DatasetTaskRef("toy", "suite", "active-cap-matrix") for _ in identities]
    specs = [
        RunSpec(
            identity, ref, "probe", ModelRoute("ADP-W-All", init_width=2, init_depth=1),
            "split.npz", str(tmp_path / f"result-{index}"), {},
            ("python", "-c", "pass"),
        )
        for index, (identity, ref) in enumerate(zip(identities, refs))
    ]
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(specs))
    settings = RunnerSettings(
        concurrency=2,
        max_active_jobs=max_active_jobs,
        max_active_gpu_jobs=max_active_gpu_jobs,
        cpu_only=False,
        run_hardness_screen=False,
        post_launch_sample_delay_seconds=0,
        pressure_poll_seconds=0.1,
    )
    resource_provider = CallbackResourceProvider(
        host_sampler=lambda: _host(4096),
        gpu_sampler=lambda _index: _gpu(4096),
        tree_sampler=lambda pid: frozenset({pid}),
        child_sampler=lambda pid, **_kwargs: ProcessMemorySample(
            host_rss_mib=100, gpu_vram_mib=100, pid_set=frozenset({pid}),
        ),
        external_host_sampler=lambda _excluded: ProcessMemorySample(),
        external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
    )
    coordinator = SuiteCoordinator(
        plan, settings, run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=resource_provider,
        wall_clock=lambda: 100.0,
    )
    active = JobRuntime(specs[0], status=RunStatus.RUNNING, device_mode="gpu")
    active.pid = 700
    active.started_at = 99.0
    active.process = SimpleNamespace(pid=700, poll=lambda: None)
    queued = JobRuntime(specs[1])
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([active], [], deque([queued])))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launches = []

    def launch(runtime, *, device, **_kwargs):
        runtime.attempt += 1
        runtime.process = SimpleNamespace(pid=701, poll=lambda: None)
        runtime.pid = 701
        runtime.device_mode = device
        runtime.started_at = 100.0
        runtime.status = RunStatus.RUNNING
        launches.append(device)
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: setattr(coordinator, "shutdown_signal", int(signal.SIGTERM))

    with pytest.raises(SystemExit) as stopped:
        coordinator.run()
    assert stopped.value.code == 130

    if max_active_jobs == 1:
        expected = None
    elif max_active_gpu_jobs == 1:
        expected = "cpu"
    else:
        expected = "gpu"
    assert launches == ([] if expected is None else [expected])


@pytest.mark.parametrize("max_active_jobs", [0, 1, 2])
@pytest.mark.parametrize("max_active_gpu_jobs", [0, 1, 2])
@pytest.mark.parametrize("host_pressure,gpu_pressure", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_full_pass_combines_active_caps_with_pressure_and_lane_fallback(
    tmp_path, monkeypatch, max_active_jobs, max_active_gpu_jobs,
    host_pressure, gpu_pressure,
):
    """Cross active caps with every hard-pressure domain in a live pass."""
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import CallbackResourceProvider, ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    identities = [
        RunIdentity(
            provider="toy", suite="suite", task_id="cap-pressure",
            split_repeat=0, fold=index, sample=0, suite_repeat=1,
            model_route=f"cap-pressure-{index}", seed=61,
            split_sha256="split", run_signature="signature",
        )
        for index in range(2)
    ]
    specs = [
        RunSpec(
            identity, DatasetTaskRef("toy", "suite", "cap-pressure"), "probe",
            ModelRoute("ADP-W-All", init_width=2, init_depth=1),
            "split.npz", str(tmp_path / f"result-{index}"), {},
            ("python", "-c", "pass"),
        )
        for index, identity in enumerate(identities)
    ]
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(specs))

    class LiveProcess:
        pid = 760

        def poll(self):
            return None

    active = JobRuntime(
        specs[0], status=RunStatus.RUNNING, device_mode="gpu", pid=760,
        process=LiveProcess(), started_at=10,
    )
    queued = JobRuntime(specs[1])
    provider = CallbackResourceProvider(
        host_sampler=lambda: _host(500 if host_pressure else 4096),
        gpu_sampler=lambda _index: _gpu(50 if gpu_pressure else 600),
        tree_sampler=lambda pid: frozenset({pid}),
        child_sampler=lambda pid, **_kwargs: ProcessMemorySample(
            host_rss_mib=100, gpu_vram_mib=100, pid_set=frozenset({pid}),
        ),
        external_host_sampler=lambda _excluded: ProcessMemorySample(),
        external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
    )
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=2, max_active_jobs=max_active_jobs,
            max_active_gpu_jobs=max_active_gpu_jobs, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=provider,
        wall_clock=lambda: 100,
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([active], [], deque([queued])))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launches = []

    def launch(runtime, *, device, **_kwargs):
        launches.append(device)
        runtime.attempt += 1
        runtime.process = SimpleNamespace(pid=761, poll=lambda: None)
        runtime.pid = 761
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: setattr(
        coordinator, "shutdown_signal", int(signal.SIGTERM),
    )

    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    if host_pressure:
        expected_device = None
    elif gpu_pressure:
        # The sole GPU child is paused; CPU remains the eligible fallback.
        expected_device = "cpu"
    elif max_active_jobs == 1:
        expected_device = None
    elif max_active_gpu_jobs == 1:
        expected_device = "cpu"
    else:
        expected_device = "gpu"
    assert launches == ([] if expected_device is None else [expected_device])
    assert active.status == (
        RunStatus.PAUSED if (host_pressure or gpu_pressure) else RunStatus.QUEUED
    )
    assert queued.device_mode == expected_device


def test_ledger_defaults_and_deliberately_unvalidated_settings_are_preserved():
    settings = RunnerSettings()
    estimator_defaults = SearchSettings()
    assert settings.concurrency == 0 and settings.effective_concurrency == 10**9
    assert (settings.max_active_jobs, settings.max_active_gpu_jobs) == (0, 0)
    assert (settings.repeat_count, settings.max_retries) == (5, 1)
    assert (settings.host_ram_stop_avail_mib, settings.host_ram_resume_avail_mib) == (1024, 1280)
    assert (settings.gpu_stop_avail_mib, settings.gpu_resume_avail_mib, settings.gpu_device_index) == (100, 500, 0)
    assert (settings.launch_host_reserve_mib, settings.launch_gpu_reserve_mib) == (0, 0)
    assert (settings.child_host_gate_buffer_mib, settings.child_gpu_gate_buffer_mib) == (500, 100)
    assert (settings.external_memory_drop_mib, settings.pressure_poll_seconds, settings.post_launch_sample_delay_seconds) == (500, 0.5, 30)
    assert settings.batch_backoff_factor == 0.5
    assert (settings.max_width, settings.max_depth, settings.max_neurons) == (512, 5, 1_000_000_000)
    assert (settings.max_epochs_vanilla, settings.max_epochs_adp) == (0, 0)
    assert settings.seed == 42 and settings.cuda_visible_devices == "0"
    assert settings.pytorch_cuda_alloc_conf == "expandable_segments:True,max_split_size_mb:128"
    assert settings.cublas_workspace_config == ":4096:8"
    for name in (
        "ex_k_width", "ex_k_depth", "patience_es", "patience_width_exp",
        "patience_depth_exp", "delta_width", "delta_depth", "max_width",
        "max_depth", "batch_size", "learning_rate",
        "scheduler_t0", "scheduler_t_mult", "scheduler_eta_min", "weight_decay",
        "grad_clip", "lambda_loss", "lambda_eq", "lambda_ineq",
        "max_candidates", "min_new_layer_width",
    ):
        assert getattr(settings, name) == getattr(estimator_defaults, name)


@pytest.mark.parametrize(
    "overrides",
    [
        {"width_only_width": 0},
        {"width_only_width": 513},
        {"width_to_depth_width": 0},
        {"width_to_depth_width": 513},
        {"width_to_depth_depth": 0},
        {"width_to_depth_depth": 6},
    ],
)
def test_ledger_rejects_invalid_initial_architecture_coordinates(overrides):
    with pytest.raises(ValueError):
        RunnerSettings(**overrides)


def test_future_model_and_dataset_plugins_plan_without_coordinator_changes(tmp_path):
    from types import SimpleNamespace

    from simple_adp.runner.datasets import DatasetTaskRef
    from simple_adp.runner.models import ModelRoute
    from simple_adp.runner.plan import SplitSpec, SuitePlan, _sha_indices

    class FutureProvider:
        name = "future-provider"

        def list_tasks(self, suite):
            assert suite == "future-suite"
            return [DatasetTaskRef(self.name, suite, "task-1")]

        def load_task(self, ref):
            return SimpleNamespace(
                X=np.zeros((4, 2), dtype=np.float32), dataset_id="dataset-1",
                name="future dataset", target_name="target", content_sha256="data-hash",
            )

        def list_splits(self, ref, data, settings):
            train, validation, test = (0, 1), (2,), (3,)
            return [SplitSpec(0, 0, 0, train, validation, test, _sha_indices(train, validation, test))]

    class FutureModelAdapter:
        name = "future-model-adapter"

        def enumerate_routes(self, generic_profiles, settings=None):
            return [ModelRoute(
                "Future-Tabular-MLP", label="future-mlp-v1", suite_group="width_only",
            )]

    settings = RunnerSettings(
        concurrency=1,
        repeat_count=1,
        missing_present_task_repeats=1,
        model_seeds=(7,),
        tasks=("width_only",),
        include_vanilla=False,
        run_hardness_screen=False,
        ex_k_width=3,
        patience_es=6,
        patience_width_exp=4,
        delta_width=0.125,
        learning_rate=0.002,
        weight_decay=0.01,
        max_candidates=9,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=tmp_path / "results",
        run_root=tmp_path / "run",
        dataset_providers=(FutureProvider(),),
        dataset_suites=(("future-provider", "future-suite"),),
        model_adapters=(FutureModelAdapter(),),
    )
    assert len(plan.runs) == 1
    assert plan.runs[0].route.model_id == "Future-Tabular-MLP"
    assert plan.runs[0].route.suite_group == "width_only"
    assert plan.runs[0].task_ref.provider == "future-provider"
    assert plan.runs[0].identity.seed != 0
    assert {
        key: plan.runs[0].worker_settings[key]
        for key in (
            "ex_k_width", "patience_es", "patience_width_exp", "delta_width",
            "learning_rate", "weight_decay", "max_candidates",
        )
    } == {
        "ex_k_width": 3, "patience_es": 6, "patience_width_exp": 4,
        "delta_width": 0.125, "learning_rate": 0.002,
        "weight_decay": 0.01, "max_candidates": 9,
    }
    assert settings.vanilla_depths == (1, 2, 3, 4, 5)
    assert settings.vanilla_widths == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    from dataclasses import replace
    changed_optimizer = replace(settings, learning_rate=0.003)
    changed_plan = SuitePlan.build(
        settings=changed_optimizer,
        results_dir=tmp_path / "other-results",
        run_root=tmp_path / "other-run",
        dataset_providers=(FutureProvider(),),
        dataset_suites=(("future-provider", "future-suite"),),
        model_adapters=(FutureModelAdapter(),),
    )
    assert changed_plan.run_signature != plan.run_signature
    assert changed_plan.runs[0].worker_settings["learning_rate"] == 0.003

    # The source intentionally leaves these controls unchecked.
    permissive = RunnerSettings(
        seed=-42, host_ram_stop_avail_mib=-5, gpu_stop_avail_mib=-2,
        swap_pressure_pct=500, swap_resume_pct=-1, job_start_index=-3,
        job_limit=-7, vanilla_min_band=-4, vanilla_max_band=-9, vanilla_samples=-2,
        repeat_count=0, max_retries=0,
    )
    assert permissive.repeat_count == 0 and permissive.max_retries == 0


def test_runner_settings_validation_boundaries():
    accepted = [
        {"seed": -42},
        {"host_ram_stop_avail_mib": -5},
        {"gpu_stop_avail_mib": -2},
        {"swap_pressure_pct": 500, "swap_resume_pct": -1},
        {"job_start_index": -3, "job_limit": -7},
        {"vanilla_min_band": -4, "vanilla_max_band": -9, "vanilla_samples": -2},
        {"repeat_count": 0, "max_retries": 0},
        {"host_ram_stop_avail_mib": float("nan"), "host_ram_resume_avail_mib": float("nan")},
        {"post_launch_sample_delay_seconds": float("nan")},
        {"external_memory_drop_mib": float("nan")},
        {"launch_host_reserve_mib": float("inf")},
    ]
    rejected = [
        {"concurrency": -1}, {"gpu_device_index": -1},
        {"pressure_poll_seconds": 0}, {"post_launch_sample_delay_seconds": -0.1},
        {"host_ram_resume_avail_mib": 100}, {"gpu_resume_avail_mib": 99},
        {"batch_backoff_factor": 0}, {"repeat_count": -1}, {"max_retries": -1},
        {"prediction_repeats": 0}, {"missing_present_task_repeats": 0},
        {"max_active_jobs": -1}, {"launch_host_reserve_mib": -1},
        {"child_gpu_gate_buffer_mib": -1}, {"external_memory_drop_mib": 0},
        {"width_only_width": 0}, {"width_to_depth_depth": 0},
        {"max_width": 0}, {"max_depth": 0}, {"max_neurons": 0},
        {"max_epochs_vanilla": -1}, {"width_only_depths": ()},
        {"width_only_depths": (6,)}, {"width_to_depth_widths": (513,)},
        {"depth_only_widths": (0,)}, {"vanilla_widths": (0,)},
    ]
    for overrides in accepted:
        RunnerSettings(**overrides)
    for overrides in rejected:
        with pytest.raises(ValueError):
            RunnerSettings(**overrides)




def test_runner_settings_preserve_documented_shared_defaults():
    settings = RunnerSettings()
    assert settings.concurrency == 0
    assert settings.max_active_jobs == 0
    assert settings.max_active_gpu_jobs == 0
    assert (settings.repeat_count, settings.max_retries) == (5, 1)
    assert (settings.missing_present_task_repeats, settings.prediction_repeats) == (1, 1)
    assert (settings.host_ram_stop_avail_mib, settings.host_ram_resume_avail_mib) == (1024.0, 1280.0)
    assert (settings.gpu_stop_avail_mib, settings.gpu_resume_avail_mib) == (100.0, 500.0)
    assert (settings.launch_host_reserve_mib, settings.launch_gpu_reserve_mib) == (0.0, 0.0)
    assert (settings.child_host_gate_buffer_mib, settings.child_gpu_gate_buffer_mib) == (500.0, 100.0)
    assert settings.external_memory_drop_mib == 500.0
    assert (settings.pressure_poll_seconds, settings.post_launch_sample_delay_seconds) == (0.5, 30.0)
    assert settings.batch_backoff_factor == 0.5
    assert settings.gpu_device_index == 0
    assert settings.cpu_only is False
    assert (settings.swap_pressure_pct, settings.swap_resume_pct) == (70.0, 70.0)
    assert settings.cuda_visible_devices == "0"
    assert settings.pytorch_cuda_alloc_conf == "expandable_segments:True,max_split_size_mb:128"
    assert (settings.shared_cpu, settings.cpu_workers) == (False, 0)
    assert (settings.max_width, settings.max_depth, settings.max_neurons) == (512, 5, 1_000_000_000)
    assert (settings.max_epochs_vanilla, settings.max_epochs_adp) == (0, 0)
    assert settings.seed == 42
    assert settings.width_only_depths == (1, 2, 3, 4, 5)
    assert settings.width_only_width == 1
    # The legacy width-to-depth list is a compatibility knob only. Job
    # generation uses the scalar seed fields and the source default list is [1].
    assert settings.width_to_depth_widths == (1,)
    assert settings.width_to_depth_width == 1
    assert settings.width_to_depth_depth == 1
    assert settings.depth_only_widths == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert settings.vanilla_depths == (1, 2, 3, 4, 5)
    assert settings.vanilla_widths == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def test_plan_enumerates_registered_routes_and_asymmetric_repeats(tmp_path):
    from simple_adp.runner.plan import _sha_indices

    class Data:
        X = np.zeros((6, 2), dtype=np.float32)
        y = np.asarray([0, 1, 0, 1, 0, 1])
        task_kind = "classification"
        dataset_id = "toy-dataset"
        content_sha256 = "data-digest"
        feature_schema = None
        name = "toy"
        target_name = "label"

    class Provider:
        name = "toy"

        def list_tasks(self, suite):
            return [DatasetTaskRef(self.name, suite, "task-1")]

        def load_task(self, _ref):
            return Data()

        def list_splits(self, ref, data, _settings):
            train, validation, test = (0, 1, 2), (3,), (4, 5)
            return [SplitSpec(0, 0, 0, train, validation, test, _sha_indices(train, validation, test))]

    provider = Provider()
    adapter = SimpleADPModelAdapter()
    settings = RunnerSettings(
        model_seeds=(0,), repeat_count=3, missing_present_task_repeats=1,
        prediction_repeats=1, include_vanilla=True,
    )
    plan = SuitePlan.build(
        settings=settings, results_dir=tmp_path / "results", run_root=tmp_path / "run",
        dataset_providers=(provider,), dataset_suites=(("toy", "suite"),),
        model_adapters=(adapter,),
    )
    # Width-only routes expand over five depths; depth-only routes over ten
    # initial widths; the eight generic IDs have two profiles; fixed routes
    # include the aggregate grid and 50 individual candidates. Adaptive routes
    # run once, 51 fixed routes run on all three outer repeats, and screening
    # contributes 22 architectures at two seeds: 51 + 153 + 44.
    assert len(plan.runs) == 248
    assert plan.manifest["effective_repeat_count"] == 3
    assert sum(spec.route.model_id == "MLP-Fixed" and not spec.route.screening_only for spec in plan.runs) == 153
    assert sum(spec.route.screening_only for spec in plan.runs) == 44
    labels = {spec.route.route_label for spec in plan.runs if spec.route.model_id == "MLP-Fixed" and not spec.route.screening_only}
    assert len(labels) == 51

    zero_repeat = SuitePlan.build(
        settings=RunnerSettings(
            model_seeds=(0,), repeat_count=0, missing_present_task_repeats=1,
            run_hardness_screen=False,
        ),
        results_dir=tmp_path / "zero-repeat-results", run_root=tmp_path / "zero-repeat-run",
        dataset_providers=(provider,), dataset_suites=(("toy", "suite"),),
        model_adapters=(adapter,),
    )
    assert len(zero_repeat.runs) == 102  # one effective pass despite configured zero
    assert zero_repeat.manifest["effective_repeat_count"] == 1
    assert {run.identity.suite_repeat for run in zero_repeat.runs} == {1}

    repeat_asymmetry = SuitePlan.build(
        settings=RunnerSettings(
            model_seeds=(0,), repeat_count=2, missing_present_task_repeats=3,
            prediction_repeats=5, run_hardness_screen=False,
        ),
        results_dir=tmp_path / "repeat-asymmetry-results", run_root=tmp_path / "repeat-asymmetry-run",
        dataset_providers=(provider,), dataset_suites=(("toy", "suite"),),
        model_adapters=(adapter,),
    )
    assert len(repeat_asymmetry.runs) == 306
    assert repeat_asymmetry.manifest["effective_repeat_count"] == 5
    assert {run.identity.suite_repeat for run in repeat_asymmetry.runs} == {1, 2, 3}

    backlog = SuitePlan.build(
        settings=RunnerSettings(model_seeds=(0,), surface="backlog", include_vanilla=True),
        results_dir=tmp_path / "backlog-results", run_root=tmp_path / "backlog-run",
        dataset_providers=(provider,), dataset_suites=(("toy", "suite"),),
        model_adapters=(adapter,),
    )
    assert backlog.runs == ()


def test_builtin_model_adapter_enumerates_all_canonical_and_vanilla_routes():
    adapter = SimpleADPModelAdapter()
    routes = adapter.enumerate_routes(("faithful", "corrected"), RunnerSettings())
    by_model = {}
    for route in routes:
        by_model.setdefault(route.model_id, []).append(route)

    assert set(by_model) == set(MODEL_REGISTRY)
    assert len(routes) == 146  # 102 model/baseline routes plus 44 diagnostic screens
    assert len(by_model["LGND-W-Greedy"]) == 5
    assert len(by_model["ADP-W-RR"]) == 5
    assert len(by_model["ADP-W-All"]) == 5
    assert len(by_model["ADP-D-D1"]) == 10
    assert len(by_model["ADP-D-Dcur"]) == 10

    mixed_ids = {
        "ADP-WD-RR-D1", "ADP-WD-All-Dcur", "ADP-WD-RR-Dcur", "ADP-WD-All-D1",
        "ADP-DW-D1-RR", "ADP-DW-Dcur-All", "ADP-DW-Dcur-RR", "ADP-DW-D1-All",
    }
    for model_id in mixed_ids:
        model_routes = by_model[model_id]
        assert {route.generic_profile for route in model_routes} == {"faithful", "corrected"}
        assert len(model_routes) == 2

    assert sum(route.screening_only for route in routes) == 44
    vanilla_routes = [route for route in by_model["MLP-Fixed"] if not route.screening_only]
    assert len(vanilla_routes) == 51
    assert sum(len(route.fixed_grid or ()) == 50 for route in vanilla_routes) == 1
    individual_architectures = {
        route.fixed_grid[0] for route in vanilla_routes if route.fixed_grid and len(route.fixed_grid) == 1
    }
    assert len(individual_architectures) == 50


def test_missing_present_task_repeats_must_be_positive():
    with pytest.raises(ValueError, match="missing_present_task_repeats must be >=1"):
        RunnerSettings(missing_present_task_repeats=0)


def test_run_seed_uses_ledger_sha256_width_modulus_and_zero_fallback(monkeypatch):
    import hashlib
    from types import SimpleNamespace

    import simple_adp.runner.plan as plan_module

    base_seed = 42
    fields = ("provider", "suite", "task", 2, "ADP-W-All")
    payload = "|".join((str(base_seed), *(str(value) for value in fields)))
    expected = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31 - 1)
    if expected == 0:
        expected = max(1, base_seed % (2**31 - 1))
    actual = plan_module.derive_run_seed(base_seed, fields)
    assert actual == expected
    assert 0 < actual < 2**31 - 1

    monkeypatch.setattr(plan_module.hashlib, "sha256", lambda _payload: SimpleNamespace(digest=lambda: bytes(32)))
    assert plan_module.derive_run_seed(42, fields) == 42


def test_coordinator_launches_one_child_and_persists_completion(tmp_path, monkeypatch):
    import sys
    import simple_adp.runner.runtime_tuning as tuning

    monkeypatch.setattr(tuning, "_apply_affinity", lambda _cpus: None)
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="probe", seed=4,
        split_sha256="split", run_signature="suite-signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    route = ModelRoute("ADP-W-All")
    spec = RunSpec(
        identity, ref, "probe", route, str(tmp_path / "split.npz"),
        str(tmp_path / "results" / "one"), {}, (sys.executable, "-c", "pass"),
    )
    plan = SuitePlan(
        "suite-signature", {"run_signature": "suite-signature", "jobs": [spec.as_dict()]}, (spec,),
    )
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(cpu_only=True, concurrency=1, post_launch_sample_delay_seconds=0,
                             pressure_poll_seconds=0.01),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, host_sampler=lambda: _host(8192), gpu_sampler=lambda _index: None,
        poll_sleep=lambda duration: time.sleep(min(duration, 0.01)),
    )
    summary = coordinator.run()
    record = coordinator.state_store.load(identity.run_id)
    assert summary["completed"] == 1
    assert record["status"] == "completed" and record["exit_code"] == 0
    assert (tmp_path / "run" / "final_report.json").is_file()


def test_hardness_screen_uses_validation_only_and_never_filters_routes(tmp_path):
    from simple_adp.runner.plan import RunSpec

    routes = []
    metrics = {
        (1, 1): [(0.80, 0.75), (0.82, 0.76)],
        (1, 2): [(0.84, 0.80), (0.86, 0.82)],
        (2, 2): [(0.90, 0.91), (0.92, 0.93)],
    }
    for (depth, width), seeds in metrics.items():
        for index, (accuracy, balanced) in enumerate(seeds):
            label = f"screen-d{depth}-w{width}-seed{index}"
            identity = RunIdentity(
                "toy", "suite", "task", 0, 0, 0, 1, label, 100 + index,
                "split", "signature",
            )
            result_dir = tmp_path / "screen-results" / label
            result_dir.mkdir(parents=True)
            (result_dir / "result.json").write_text(json.dumps({
                "status": "completed", "screening_only": True, "task_kind": "classification",
                "validation_metrics": {"accuracy": accuracy, "balanced_accuracy": balanced},
                "test_metrics": {"accuracy": 1.0, "balanced_accuracy": 1.0},
            }))
            routes.append(RunSpec(
                identity, DatasetTaskRef("toy", "suite", "task"), "simple_adp",
                ModelRoute(
                    "MLP-Fixed", fixed_grid=(tuple([width] * depth),), label=label,
                    screening_only=True, screening_seed=100 + index,
                    screening_architecture=(depth, width),
                ), str(tmp_path / "split.npz"), str(result_dir), {}, ("unused",),
            ))
    report = build_screening_reports(
        routes, tmp_path / "screen-results", classification_ceiling=1.0,
        regression_ceiling_r2=0.999, min_complexity_gain=0.02,
        min_hard_fold_fraction=0.8, screen_seed_count=2,
    )
    fold = json.loads((tmp_path / "screen-results" / "screening-folds.jsonl").read_text())
    assert fold["eligible_for_full_comparison"]
    assert fold["validation_complexity_gain"] > 0.02
    assert report["routes_filtered"] is False
    assert report["metrics_source"] == "validation_only"
    assert report["eligible_folds"] == 1


def test_hardness_screen_routes_are_additive_and_report_cannot_filter_plan(tmp_path):
    from simple_adp.runner.plan import _sha_indices

    class Data:
        task_kind = "classification"

    class Provider:
        name = "toy"

        def list_tasks(self, suite):
            return [DatasetTaskRef("toy", suite, "task-1")]

        def load_task(self, _ref):
            return Data()

        def list_splits(self, _ref, _data, _settings):
            train, validation, test = (0, 1), (2,), (3,)
            return [SplitSpec(0, 0, 0, train, validation, test, _sha_indices(train, validation, test))]

    provider = Provider()
    adapter = SimpleADPModelAdapter()
    common = RunnerSettings(model_seeds=(0,), repeat_count=1, missing_present_task_repeats=1)
    without_screen = SuitePlan.build(
        settings=replace(common, run_hardness_screen=False),
        results_dir=tmp_path / "without-screen-results",
        run_root=tmp_path / "without-screen-run",
        dataset_providers=(provider,), dataset_suites=(("toy", "suite"),),
        model_adapters=(adapter,),
    )
    with_screen = SuitePlan.build(
        settings=replace(common, run_hardness_screen=True),
        results_dir=tmp_path / "with-screen-results",
        run_root=tmp_path / "with-screen-run",
        dataset_providers=(provider,), dataset_suites=(("toy", "suite"),),
        model_adapters=(adapter,),
    )

    def experiment_route(spec):
        return (
            spec.route.model_id, spec.route.generic_profile, spec.route.route_label,
            spec.route.suite_group, spec.route.fixed_grid, spec.identity.suite_repeat,
        )

    expected = [experiment_route(spec) for spec in without_screen.runs]
    actual = [experiment_route(spec) for spec in with_screen.runs if not spec.route.screening_only]
    screens = [spec for spec in with_screen.runs if spec.route.screening_only]
    assert actual == expected
    assert len(screens) == 44

    report = build_screening_reports(
        with_screen.runs, tmp_path / "with-screen-results",
        classification_ceiling=1.0, regression_ceiling_r2=0.999,
        min_complexity_gain=0.02, min_hard_fold_fraction=0.8,
        screen_seed_count=common.screen_seed_count,
    )
    assert report["routes_filtered"] is False
    assert report["eligible_datasets"] == 0  # no screening result can make this plan eligible
    assert [experiment_route(spec) for spec in with_screen.runs if not spec.route.screening_only] == expected


def test_estimator_resumes_interrupted_stage_exactly(tmp_path, monkeypatch):
    class InterruptOnceRegression(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            if self.interrupt and self.calls == 2:
                self.interrupt = False
                raise RuntimeError("simulated worker interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(119)
    X = rng.normal(size=(12, 3)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 2]).astype(np.float32)
    settings = SearchSettings(
        init_width=3, init_depth=2, max_width=4, max_depth=3, max_neurons=64,
        max_epochs=2, patience_es=2, patience_width_exp=0, batch_size=4, seed=5,
    )
    checkpoint = tmp_path / "resume.pt"
    interrupted = TabularADP("ADP-W-All", task=InterruptOnceRegression(True), settings=settings)
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        interrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)

    resumed = TabularADP("ADP-W-All", task=interrupted.task, settings=settings)
    resumed.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)
    uninterrupted = TabularADP("ADP-W-All", task=RegressionTask(), settings=settings)
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))
    assert resumed.history_ == uninterrupted.history_
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))

    settings_changed = SearchSettings(**{**settings.as_dict(), "seed": settings.seed + 1})
    incompatible = TabularADP("ADP-W-All", task=RegressionTask(), settings=settings_changed)
    with pytest.raises(ValueError, match="signature does not match"):
        incompatible.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)

    # Registry IDs are stable API names, but they are not enough to identify
    # the mechanism if a registry entry is later remapped in code.
    changed_mechanism = replace(MODEL_REGISTRY["ADP-W-All"], width_strategy="round_robin")
    monkeypatch.setattr(estimator_module, "get_model_config", lambda _model_id: changed_mechanism)
    mechanism_incompatible = TabularADP("ADP-W-All", task=RegressionTask(), settings=settings)
    with pytest.raises(ValueError, match="signature does not match"):
        mechanism_incompatible.fit(
            X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint,
        )


def test_generic_search_resumes_candidate_stage_and_continues(tmp_path):
    class InterruptCandidateRegression(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            if self.interrupt and self.calls == 4:
                self.interrupt = False
                raise RuntimeError("simulated candidate interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(221)
    X = rng.normal(size=(12, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1]).astype(np.float32)
    settings = SearchSettings(
        init_width=2, init_depth=1, ex_k_width=1, max_width=4, max_depth=2,
        max_neurons=64, max_epochs=1, patience_es=1, patience_width_exp=2,
        patience_depth_exp=0, max_candidates=1, batch_size=4, seed=31,
    )
    checkpoint = tmp_path / "generic-resume.pt"
    interrupted = TabularADP("ADP-WD-RR-D1", task=InterruptCandidateRegression(True), settings=settings)
    with pytest.raises(RuntimeError, match="simulated candidate interruption"):
        interrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)

    resumed = TabularADP("ADP-WD-RR-D1", task=interrupted.task, settings=settings)
    resumed.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)
    uninterrupted = TabularADP("ADP-WD-RR-D1", task=RegressionTask(), settings=settings)
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))
    assert resumed.history_ == uninterrupted.history_
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
@pytest.mark.parametrize("model_id", [
    "ADP-WD-RR-D1", "ADP-WD-All-Dcur", "ADP-WD-RR-Dcur", "ADP-WD-All-D1",
    "ADP-DW-D1-RR", "ADP-DW-Dcur-All", "ADP-DW-Dcur-RR", "ADP-DW-D1-All",
])
def test_generic_mixed_order_routes_resume_candidate_stage_exactly(
    tmp_path, profile, model_id,
):
    """Every mixed-order mechanism resumes an interrupted candidate in place."""
    class InterruptCandidateRegression(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            # The initial architecture is complete; interrupt during the first
            # proposed candidate, after its first batch has been checkpointed.
            if self.interrupt and self.calls == 4:
                self.interrupt = False
                raise RuntimeError("simulated mixed-order candidate interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(3301)
    X = rng.normal(size=(10, 3)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] - X[:, 2]).astype(np.float32)
    cfg = SearchSettings(
        init_width=2, init_depth=2, ex_k_width=1, max_width=4, max_depth=4,
        max_neurons=128, max_epochs=1, patience_es=2,
        patience_width_exp=1, patience_depth_exp=1, max_candidates=1,
        batch_size=4, seed=47,
    )
    checkpoint = tmp_path / f"{model_id}-{profile}.pt"
    task = InterruptCandidateRegression(True)
    interrupted = TabularADP(
        model_id, task=task, settings=cfg, generic_profile=profile,
    )
    with pytest.raises(RuntimeError, match="simulated mixed-order candidate interruption"):
        interrupted.fit(
            X[:8], y[:8], validation_data=(X[8:], y[8:]),
            checkpoint_path=checkpoint,
        )

    resumed = TabularADP(
        model_id, task=task, settings=cfg, generic_profile=profile,
    )
    resumed.fit(
        X[:8], y[:8], validation_data=(X[8:], y[8:]),
        checkpoint_path=checkpoint,
    )
    uninterrupted = TabularADP(
        model_id, task=RegressionTask(), settings=cfg, generic_profile=profile,
    )
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))

    assert resumed.history_ == uninterrupted.history_
    assert resumed.search_state_["candidate_count"] == 1
    assert resumed.search_state_["profile"] == profile
    assert resumed.search_state_["current_architecture"] == uninterrupted.search_state_["current_architecture"]
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
@pytest.mark.parametrize("mutation_kind", ["width", "depth"])
def test_custom_generic_mutation_resumes_candidate_without_reapplying_mutation(
    tmp_path, profile, mutation_kind,
):
    from simple_adp import TabularMLP

    class InterruptCandidateRegression(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            if self.interrupt and self.calls == 4:
                self.interrupt = False
                raise RuntimeError("simulated custom-mutation candidate interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(8201)
    X = rng.normal(size=(10, 3)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] - X[:, 2]).astype(np.float32)
    width_settings = mutation_kind == "width"
    settings = SearchSettings(
        init_width=2, init_depth=2, ex_k_width=1,
        max_width=5 if width_settings else 2, max_depth=3,
        max_neurons=64, max_epochs=1, patience_es=2,
        patience_width_exp=2, patience_depth_exp=1 if not width_settings else 0,
        max_candidates=2 if width_settings else 1, batch_size=4, seed=89,
    )
    mutation_calls = []

    def widen(model, _settings):
        mutation_calls.append(("width", tuple(model.hidden_widths)))
        return TabularMLP(
            model.input_dim,
            [width + 1 for width in model.hidden_widths],
            model.output_dim,
        )

    def deepen(model, _settings):
        mutation_calls.append(("depth", tuple(model.hidden_widths)))
        return TabularMLP(
            model.input_dim, [*model.hidden_widths, 1], model.output_dim,
        )

    widen_fn = widen if width_settings else None
    deepen_fn = deepen if not width_settings else None

    checkpoint = tmp_path / f"custom-mutation-{profile}.pt"
    task = InterruptCandidateRegression(True)
    interrupted = TabularADP(
        "ADP-WD-RR-D1", task=task, settings=settings, generic_profile=profile,
        generic_widen_fn=widen_fn, generic_deepen_fn=deepen_fn,
        generic_mutation_id=f"custom-{mutation_kind}-policy-v1",
    )
    with pytest.raises(RuntimeError, match="simulated custom-mutation candidate interruption"):
        interrupted.fit(
            X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint,
        )

    # The mutation has already materialized the active candidate. Resume must
    # restore its tensors and optimizer state instead of asking arbitrary user
    # callback code to construct that candidate again.
    calls_at_interruption = list(mutation_calls)
    resumed = TabularADP(
        "ADP-WD-RR-D1", task=task, settings=settings, generic_profile=profile,
        generic_widen_fn=widen_fn, generic_deepen_fn=deepen_fn,
        generic_mutation_id=f"custom-{mutation_kind}-policy-v1",
    )
    resumed.fit(
        X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint,
    )
    assert mutation_calls[:len(calls_at_interruption)] == calls_at_interruption
    assert len(mutation_calls) == len(calls_at_interruption) + int(width_settings)

    uninterrupted_calls = []

    def uninterrupted_widen(model, _settings):
        uninterrupted_calls.append(("width", tuple(model.hidden_widths)))
        return TabularMLP(
            model.input_dim,
            [width + 1 for width in model.hidden_widths],
            model.output_dim,
        )

    def uninterrupted_deepen(model, _settings):
        uninterrupted_calls.append(("depth", tuple(model.hidden_widths)))
        return TabularMLP(
            model.input_dim, [*model.hidden_widths, 1], model.output_dim,
        )

    uninterrupted = TabularADP(
        "ADP-WD-RR-D1", task=RegressionTask(), settings=settings,
        generic_profile=profile,
        generic_widen_fn=uninterrupted_widen if width_settings else None,
        generic_deepen_fn=uninterrupted_deepen if not width_settings else None,
        generic_mutation_id=f"custom-{mutation_kind}-policy-v1",
    )
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))

    assert mutation_calls == uninterrupted_calls
    assert resumed.history_ == uninterrupted.history_
    assert resumed.search_state_ == uninterrupted.search_state_
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
def test_custom_task_adapter_resumes_generic_candidate_stage(tmp_path, profile):
    from simple_adp import CallableTask

    class InterruptCallableTask(CallableTask):
        def __init__(self, interrupt):
            self.interrupt = interrupt
            self.loss_calls = 0

            def loss_fn(outputs, targets):
                return torch.nn.functional.l1_loss(outputs, targets.reshape(-1, 1))

            def score_fn(outputs, targets):
                return float(loss_fn(outputs, targets))

            super().__init__(
                1, loss_fn, score_fn, lambda outputs: outputs[:, 0], task_name="resume-mae",
            )

        def loss(self, outputs, targets):
            self.loss_calls += 1
            if self.interrupt and self.loss_calls == 4:
                self.interrupt = False
                raise RuntimeError("simulated custom-task candidate interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(9107)
    X = rng.normal(size=(10, 3)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 2]).astype(np.float32)
    settings = SearchSettings(
        init_width=2, init_depth=2, ex_k_width=1, max_width=4, max_depth=3,
        max_neurons=64, max_epochs=1, patience_es=2, patience_width_exp=1,
        patience_depth_exp=0, max_candidates=1, batch_size=4, seed=97,
    )
    checkpoint = tmp_path / f"custom-task-{profile}.pt"
    task = InterruptCallableTask(True)
    interrupted = TabularADP(
        "ADP-WD-RR-D1", task=task, settings=settings, generic_profile=profile,
    )
    with pytest.raises(RuntimeError, match="simulated custom-task candidate interruption"):
        interrupted.fit(
            X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint,
        )

    resumed = TabularADP(
        "ADP-WD-RR-D1", task=InterruptCallableTask(False), settings=settings,
        generic_profile=profile,
    )
    resumed.fit(
        X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint,
    )

    uninterrupted_task = CallableTask(
        1,
        lambda outputs, targets: torch.nn.functional.l1_loss(outputs, targets.reshape(-1, 1)),
        lambda outputs, targets: float(
            torch.nn.functional.l1_loss(outputs, targets.reshape(-1, 1))
        ),
        lambda outputs: outputs[:, 0],
        task_name="resume-mae",
    )
    uninterrupted = TabularADP(
        "ADP-WD-RR-D1", task=uninterrupted_task, settings=settings,
        generic_profile=profile,
    )
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))

    assert resumed.history_ == uninterrupted.history_
    assert resumed.search_state_ == uninterrupted.search_state_
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
def test_generic_width_to_depth_resume_preserves_phase_transition(tmp_path, profile):
    class ScriptedInterruptedTask(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.loss_calls = 0
            self.validation_calls = 0

        def loss(self, outputs, targets):
            self.loss_calls += 1
            if self.interrupt and self.loss_calls == 3:
                self.interrupt = False
                raise RuntimeError("simulated width-to-depth transition interruption")
            return super().loss(outputs, targets)

        def validation_score(self, outputs, targets):
            scores = (1.0, 1.2, 0.4)
            score = scores[self.validation_calls]
            self.validation_calls += 1
            return score

    rng = np.random.default_rng(1107)
    X = rng.normal(size=(10, 2)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] ** 2).astype(np.float32)
    settings = SearchSettings(
        init_width=2, init_depth=2, ex_k_width=1, max_width=4, max_depth=3,
        max_neurons=64, max_epochs=1, patience_es=1, patience_width_exp=1,
        patience_depth_exp=1, max_candidates=0, batch_size=8, seed=109,
    )
    checkpoint = tmp_path / f"width-to-depth-{profile}.pt"
    interrupted = TabularADP(
        "ADP-WD-All-Dcur", task=ScriptedInterruptedTask(True),
        settings=settings, generic_profile=profile,
    )
    with pytest.raises(RuntimeError, match="width-to-depth transition interruption"):
        interrupted.fit(
            X[:8], y[:8], validation_data=(X[8:], y[8:]),
            checkpoint_path=checkpoint,
        )

    resumed = TabularADP(
        "ADP-WD-All-Dcur", task=interrupted.task,
        settings=settings, generic_profile=profile,
    )
    resumed.fit(
        X[:8], y[:8], validation_data=(X[8:], y[8:]),
        checkpoint_path=checkpoint,
    )
    uninterrupted = TabularADP(
        "ADP-WD-All-Dcur", task=ScriptedInterruptedTask(False),
        settings=settings, generic_profile=profile,
    )
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))

    assert [row["stage"] for row in resumed.history_] == ["initial", "width", "depth"]
    assert resumed.history_ == uninterrupted.history_
    assert resumed.search_state_["candidate_count"] == 2
    assert resumed.search_state_["profile"] == profile
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
def test_depth_then_width_resume_preserves_global_candidate_ceiling(tmp_path, profile):
    class InterruptDepthCandidate(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            # Fail on the first loss call of the depth candidate, after the initial
            # one-epoch full-batch stage has completed.
            if self.interrupt and self.calls == 2:
                self.interrupt = False
                raise RuntimeError("simulated depth-candidate interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(7301)
    X = rng.normal(size=(10, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 1]).astype(np.float32)
    settings = SearchSettings(
        init_width=2, init_depth=2, ex_k_width=1, max_width=4, max_depth=4,
        max_neurons=64, max_epochs=1, patience_es=2,
        patience_width_exp=1, patience_depth_exp=1, max_candidates=1,
        batch_size=8, seed=73,
    )
    checkpoint = tmp_path / f"depth-width-cap-{profile}.pt"
    interrupted = TabularADP(
        "ADP-DW-D1-RR", task=InterruptDepthCandidate(True), settings=settings,
        generic_profile=profile,
    )
    with pytest.raises(RuntimeError, match="simulated depth-candidate interruption"):
        interrupted.fit(
            X[:8], y[:8], validation_data=(X[8:], y[8:]),
            checkpoint_path=checkpoint,
        )

    resumed = TabularADP(
        "ADP-DW-D1-RR", task=interrupted.task, settings=settings,
        generic_profile=profile,
    )
    resumed.fit(
        X[:8], y[:8], validation_data=(X[8:], y[8:]),
        checkpoint_path=checkpoint,
    )
    uninterrupted = TabularADP(
        "ADP-DW-D1-RR", task=RegressionTask(), settings=settings,
        generic_profile=profile,
    )
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))

    assert resumed.history_ == uninterrupted.history_
    assert [row["stage"] for row in resumed.history_] == ["initial", "depth"]
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


def test_round_robin_growth_cursor_survives_estimator_resume(tmp_path):
    class InterruptFirstCandidate(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            if self.interrupt and self.calls == 4:
                self.interrupt = False
                raise RuntimeError("simulated round-robin interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(902)
    X = rng.normal(size=(12, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)
    settings = SearchSettings(
        init_width=2, init_depth=2, ex_k_width=1, max_width=5, max_depth=3,
        max_neurons=64, max_epochs=1, patience_es=1, patience_width_exp=3,
        patience_depth_exp=0, max_candidates=2, batch_size=4, seed=83,
    )
    checkpoint = tmp_path / "cursor-resume.pt"
    interrupted = TabularADP("ADP-WD-RR-D1", task=InterruptFirstCandidate(True), settings=settings)
    with pytest.raises(RuntimeError, match="round-robin interruption"):
        interrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)

    resumed = TabularADP("ADP-WD-RR-D1", task=interrupted.task, settings=settings)
    resumed.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)
    uninterrupted = TabularADP("ADP-WD-RR-D1", task=RegressionTask(), settings=settings)
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))
    assert resumed.history_ == uninterrupted.history_
    assert resumed.history_[-1]["architecture"] == [3, 3]


def test_dedicated_width_candidate_resume_preserves_search_cursor(tmp_path):
    class InterruptWidthRegression(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            if self.interrupt and self.calls == 4:
                self.interrupt = False
                raise RuntimeError("simulated width interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(315)
    X = rng.normal(size=(12, 2)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] ** 2).astype(np.float32)
    settings = SearchSettings(
        init_width=2, init_depth=1, ex_k_width=1, max_width=4, max_depth=2,
        max_neurons=64, max_epochs=1, patience_es=1, patience_width_exp=1,
        patience_depth_exp=0, batch_size=4, seed=43,
    )
    checkpoint = tmp_path / "width-resume.pt"
    interrupted = TabularADP("ADP-W-All", task=InterruptWidthRegression(True), settings=settings)
    with pytest.raises(RuntimeError, match="width interruption"):
        interrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)

    resumed = TabularADP("ADP-W-All", task=interrupted.task, settings=settings)
    resumed.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)
    uninterrupted = TabularADP("ADP-W-All", task=RegressionTask(), settings=settings)
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))
    assert resumed.history_ == uninterrupted.history_
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


def test_fixed_grid_resumes_current_candidate_and_remaining_grid(tmp_path):
    class InterruptFixedRegression(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            if self.interrupt and self.calls == 4:
                self.interrupt = False
                raise RuntimeError("simulated fixed-grid interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(412)
    X = rng.normal(size=(12, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] ** 2).astype(np.float32)
    settings = SearchSettings(
        init_width=2, init_depth=1, max_width=4, max_depth=2, max_neurons=64,
        max_epochs=1, patience_es=1, patience_width_exp=0, batch_size=4, seed=53,
    )
    grid = [[2], [3]]
    checkpoint = tmp_path / "fixed-resume.pt"
    interrupted = TabularADP("MLP-Fixed", task=InterruptFixedRegression(True), settings=settings, fixed_grid=grid)
    with pytest.raises(RuntimeError, match="simulated fixed-grid interruption"):
        interrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)

    resumed = TabularADP("MLP-Fixed", task=interrupted.task, settings=settings, fixed_grid=grid)
    resumed.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)
    uninterrupted = TabularADP("MLP-Fixed", task=RegressionTask(), settings=settings, fixed_grid=grid)
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))
    assert resumed.history_ == uninterrupted.history_
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


@pytest.mark.parametrize("initial_width", [2, 3])
def test_progressive_depth_growth_resumes_at_every_stage_boundary(
    tmp_path, initial_width,
):
    # Each stage has two training batches and one validation loss call. Cover
    # every interruption point after the initial stage completes, through every
    # progressive catch-up stage, and through the completed-width candidate.
    last_stage_loss_call = 3 * (initial_width + 1)
    interrupt_calls = range(4, last_stage_loss_call + 1)

    for interrupt_at in interrupt_calls:
        _assert_progressive_depth_resume_matches_uninterrupted(
            tmp_path / f"width-{initial_width}-call-{interrupt_at}",
            initial_width=initial_width,
            interrupt_at=interrupt_at,
        )


def _assert_progressive_depth_resume_matches_uninterrupted(
    checkpoint_dir, *, initial_width, interrupt_at,
):
    class InterruptCatchupRegression(RegressionTask):
        def __init__(self, interrupt_at):
            super().__init__()
            self.interrupt_at = interrupt_at
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            if self.calls == self.interrupt_at:
                raise RuntimeError("simulated progressive catch-up interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(932)
    X = rng.normal(size=(12, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] - X[:, 1]).astype(np.float32)
    settings = SearchSettings(
        init_width=initial_width, init_depth=1, max_width=3, max_depth=3, max_neurons=64,
        max_epochs=1, patience_es=1, patience_width_exp=0, patience_depth_exp=1,
        batch_size=4, seed=67,
    )
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "progressive-resume.pt"
    interrupted = TabularADP(
        "ADP-D-D1", task=InterruptCatchupRegression(interrupt_at), settings=settings,
    )
    with pytest.raises(RuntimeError, match="progressive catch-up interruption"):
        interrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)

    interrupted.task.interrupt_at = -1
    resumed = TabularADP("ADP-D-D1", task=interrupted.task, settings=settings)
    resumed.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)
    uninterrupted = TabularADP("ADP-D-D1", task=RegressionTask(), settings=settings)
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))
    assert resumed.history_ == uninterrupted.history_
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


@pytest.mark.parametrize("model_id", ["ADP-D-D1", "ADP-D-Dcur"])
def test_depth_candidate_resume_matches_uninterrupted_search(tmp_path, model_id):
    class InterruptDepthRegression(RegressionTask):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.calls = 0

        def loss(self, outputs, targets):
            self.calls += 1
            if self.interrupt and self.calls == 4:
                self.interrupt = False
                raise RuntimeError("simulated completed depth candidate interruption")
            return super().loss(outputs, targets)

    rng = np.random.default_rng(173)
    X = rng.normal(size=(12, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 1] ** 2).astype(np.float32)
    settings = SearchSettings(
        init_width=2, init_depth=2, max_width=4, max_depth=3, max_neurons=64,
        max_epochs=1, patience_es=1, patience_width_exp=0, patience_depth_exp=1,
        batch_size=4, seed=79,
    )
    checkpoint = tmp_path / f"{model_id}-candidate-resume.pt"
    interrupted = TabularADP(
        model_id, task=InterruptDepthRegression(True), settings=settings,
    )
    with pytest.raises(RuntimeError, match="completed depth candidate interruption"):
        interrupted.fit(
            X[:8], y[:8], validation_data=(X[8:], y[8:]),
            checkpoint_path=checkpoint,
        )

    resumed = TabularADP(model_id, task=interrupted.task, settings=settings)
    resumed.fit(
        X[:8], y[:8], validation_data=(X[8:], y[8:]),
        checkpoint_path=checkpoint,
    )
    uninterrupted = TabularADP(model_id, task=RegressionTask(), settings=settings)
    uninterrupted.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))
    assert resumed.history_ == uninterrupted.history_
    assert resumed.search_state_["current_architecture"] == uninterrupted.search_state_["current_architecture"]
    np.testing.assert_array_equal(resumed.predict(X[8:]), uninterrupted.predict(X[8:]))


def test_launch_error_retries_exhaust_then_finalize_without_aborting_suite(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.state import RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="launch-failure", seed=4,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(tmp_path / "result"), {}, ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": [spec.as_dict()]}, (spec,))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(
            concurrency=1, max_retries=1, cpu_only=True,
            run_hardness_screen=False, post_launch_sample_delay_seconds=0,
        ), run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(_host(), None, 0, 0, 0, 0, 0, 0, 0, 0,
                                frozenset(), frozenset(), False, False, 1)
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "choose_device", lambda *_args, **_kwargs: "cpu")
    launch_attempts = []

    def fail_launch(*_args, **_kwargs):
        launch_attempts.append(1)
        raise OSError("synthetic Popen failure")

    monkeypatch.setattr(scheduler_module, "launch_job", fail_launch)
    summary = coordinator.run()
    record = coordinator.state_store.load(identity.run_id)
    assert len(launch_attempts) == 2
    assert record["status"] == RunStatus.FAILED.value
    assert record["last_error"] == "launch_error"
    assert summary["failed_attempts"] == 2
    assert summary["terminal_failures"] == 1


def test_launch_failures_retry_at_queue_tail_inside_fill_loop(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import RunStatus

    def make_spec(route_name):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=route_name, seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        return RunSpec(
            identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
            str(tmp_path / route_name), {}, ("python", "-c", "pass"),
        )

    first, second = make_spec("first"), make_spec("second")
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (first, second))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(
            concurrency=2, max_retries=1, cpu_only=True, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1,
        ), run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, poll_sleep=lambda _seconds: None,
    )
    snapshot = ResourceSnapshot(
        _host(8192), None, 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 2,
    )
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "choose_device", lambda *_args, **_kwargs: "cpu")
    launch_order = []
    pass_events = []
    coordinator.sleep = lambda seconds: pass_events.append(("sleep", seconds))

    class Exited:
        pid = 99

        def poll(self):
            return 0

    def launch(runtime, **_kwargs):
        launch_order.append(runtime.spec.identity.model_route)
        pass_events.append(("launch", runtime.spec.identity.model_route))
        if runtime.spec.identity.model_route == "first" and launch_order.count("first") == 1:
            raise OSError("first launch fails once")
        runtime.process = Exited()
        runtime.pid = runtime.process.pid
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = "cpu"
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    summary = coordinator.run()

    assert launch_order == ["first", "second", "first"]
    assert pass_events[:2] == [("launch", "first"), ("launch", "second")]
    assert not any(event == ("sleep", 0) or event == ("sleep", 0.0) for event in pass_events)
    assert summary["completed"] == 2
    assert summary["retries"] == 1
    assert coordinator.counters.launched == 2


def test_uncapped_launch_failures_retry_immediately_until_a_launch_succeeds(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="uncapped-launch", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"), {},
        ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(
            concurrency=1, max_retries=0, cpu_only=True, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1,
        ), run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(
        _host(8192), None, 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 1,
    )
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "choose_device", lambda *_args, **_kwargs: "cpu")
    events = []

    class Exited:
        pid = 717

        def poll(self):
            return 0

    launch_count = 0

    def launch(runtime, **_kwargs):
        nonlocal launch_count
        launch_count += 1
        runtime.attempt += 1
        events.append(("launch", launch_count))
        if launch_count < 3:
            raise OSError("transient synthetic process creation failure")
        runtime.process = Exited()
        runtime.pid = runtime.process.pid
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = "cpu"
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda seconds: events.append(("sleep", seconds))

    summary = coordinator.run()

    assert [event for event in events if event[0] == "launch"] == [
        ("launch", 1), ("launch", 2), ("launch", 3),
    ]
    # Failed Popen attempts continue within queue filling; the controller
    # does not yield to its 0.1-second poll sleep between those retries.
    third_launch = events.index(("launch", 3))
    assert events[:third_launch] == [("launch", 1), ("launch", 2)]
    assert summary["completed"] == 1
    assert summary["failed_attempts"] == 2
    assert summary["retries"] == 2
    assert coordinator.counters.launched == 1


def test_full_pass_reaps_failed_child_retries_to_queue_tail_before_fifo_admission(tmp_path, monkeypatch):
    """An exited active job retries behind work already waiting in the FIFO."""
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    def make_spec(route_name):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=route_name, seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        return RunSpec(
            identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
            str(tmp_path / route_name), {}, ("python", "-c", "pass"),
        )

    already_active_spec, queued_first_spec, queued_second_spec = map(
        make_spec, ("active-failed", "queued-first", "queued-second")
    )
    failed = JobRuntime(
        already_active_spec, status=RunStatus.RUNNING, attempt=1,
        device_mode="cpu", pid=101, process=SimpleNamespace(pid=101, poll=lambda: 23),
    )
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []},
        (already_active_spec, queued_first_spec, queued_second_spec),
    )
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=1, max_retries=1, cpu_only=True, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(
        _host(8192), None, 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 1,
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: (
        [failed], [], deque(JobRuntime(spec) for spec in (queued_first_spec, queued_second_spec)),
    ))
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(coordinator, "_classify_failure", lambda _runtime, code: f"nonzero_exit:{code}")
    launch_order = []

    def launch(runtime, *, device, **_kwargs):
        launch_order.append(runtime.spec.identity.model_route)
        runtime.attempt += 1
        runtime.process = SimpleNamespace(pid=200 + len(launch_order), poll=lambda: 0)
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: None

    summary = coordinator.run()

    assert launch_order == ["queued-first", "queued-second", "active-failed"]
    assert summary["completed"] == 3
    assert summary["failed_attempts"] == 1
    assert summary["retries"] == 1
    recovered = coordinator.state_store.load(failed.job_id)
    assert recovered["status"] == RunStatus.COMPLETED.value
    assert recovered["attempt"] == 2
    assert recovered["retry_count"] == 1


def test_full_pass_requeues_disappeared_recovered_child_as_unknown_exit_at_fifo_tail(tmp_path, monkeypatch, caplog):
    """Recovered PIDs have no trustworthy exit code and re-enter behind queued work."""
    import logging
    import simple_adp.runner.scheduler as scheduler_module
    import simple_adp.runner.state as state_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    def make_runtime(label, *, recovered=False, process_exit=None):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="recovered", split_repeat=0,
            fold=0, sample=0, suite_repeat=1, model_route=label, seed=4,
            split_sha256="split", run_signature="signature",
        )
        spec = RunSpec(
            identity, DatasetTaskRef("toy", "suite", "recovered"), "probe",
            ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / label), {},
            ("python", "-c", "pass"),
        )
        if recovered:
            process = SimpleNamespace(pid=71, poll=lambda: 1)
        else:
            process = SimpleNamespace(pid=72, poll=lambda: process_exit)
        runtime = JobRuntime(
            spec, status=RunStatus.RUNNING, attempt=1, device_mode="cpu",
            pid=process.pid, process=process, recovered_process=recovered,
        )
        return runtime

    recovered = make_runtime("recovered", recovered=True)
    completed = make_runtime("already-finished", process_exit=0)
    queued_spec = make_runtime("queued-first").spec
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []},
        (recovered.spec, completed.spec, queued_spec),
    )
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=1, cpu_only=True, max_retries=1,
            post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1,
            run_hardness_screen=False,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(
        _host(8192), None, 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 1,
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: (
        [recovered, completed], [], deque([JobRuntime(queued_spec)]),
    ))
    monkeypatch.setattr(state_module, "_recovered_pid_live", lambda _pid: False)
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    launch_order = []

    def launch(runtime, *, device, **_kwargs):
        launch_order.append(runtime.spec.identity.model_route)
        runtime.attempt += 1
        runtime.process = SimpleNamespace(pid=100 + len(launch_order), poll=lambda: 0)
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        runtime.recovered_process = False
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: None

    with caplog.at_level(logging.WARNING, logger="simple_adp.runner"):
        summary = coordinator.run()

    assert launch_order == ["queued-first", "recovered"]
    assert summary["completed"] == 3
    assert summary["failed_attempts"] == 0
    assert summary["retries"] == 0
    assert summary["recovered_unknown_exit"] == 1
    recovered_record = coordinator.state_store.load(recovered.job_id)
    assert recovered_record["status"] == RunStatus.COMPLETED.value
    assert recovered_record["attempt"] == 2
    assert recovered_record["last_error"] is None
    events = coordinator.state_store.events_path.read_text(encoding="utf-8")
    assert "recovered_child_exit_status_unknown" in events
    assert (
        f"[RECOVER] child disappeared after orchestrator restart; requeued job={recovered.job_id} "
        "for checkpointed continuation"
    ) in [record.getMessage() for record in caplog.records]


def test_full_pass_resumes_every_ready_job_before_caps_gates_and_holds(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    specs = []
    for route_name in ("paused-one", "paused-two"):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=route_name, seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        specs.append(RunSpec(
            identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
            str(tmp_path / route_name), {}, ("python", "-c", "pass"),
        ))
    paused = [JobRuntime(
        spec, status=RunStatus.PAUSED, device_mode="gpu", pause_reason="gpu_memory_pressure",
    ) for spec in specs]
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(specs))
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=5, max_active_jobs=1, max_active_gpu_jobs=1,
            post_launch_sample_delay_seconds=30, run_hardness_screen=False,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    coordinator.admission.gpu_gate_closed = True
    coordinator.admission.host_gate_closed = True
    coordinator.admission.gpu_hold_until = 999
    coordinator.admission.cpu_hold_until = 999
    coordinator.admission.batch_scale = 0.2
    snapshot = ResourceSnapshot(
        _host(8192), _gpu(8192), 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 1,
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([], list(paused), deque()))
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(coordinator, "_resume_ready", lambda *_args: True)
    launches = []

    def launch(runtime, *, device, launch_budget, **_kwargs):
        launches.append((runtime.job_id, device, launch_budget))
        runtime.attempt += 1
        runtime.process = SimpleNamespace(pid=500 + len(launches), poll=lambda: 0)
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: None

    summary = coordinator.run()

    assert [(device, budget) for _job_id, device, budget in launches] == [("gpu", 5), ("gpu", 5)]
    assert summary["completed"] == 2


def test_full_pass_combines_gpu_oom_retry_peer_pause_and_hard_pressure_pause(tmp_path, monkeypatch):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample, ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    runtimes = {}
    for name, device, depth, exit_code in (
        ("failed-oom", "gpu", 1, 9),
        ("oom-peer", "gpu", 5, None),
        ("pressure-peer", "gpu", 3, None),
        ("cpu-worker", "cpu", 8, None),
        ("waiting", "gpu", 1, None),
    ):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=depth,
            sample=depth, suite_repeat=1, model_route=name, seed=4,
            split_sha256="split", run_signature="signature",
        )
        spec = RunSpec(
            identity, DatasetTaskRef("toy", "suite", "1"), "probe",
            ModelRoute("ADP-W-All", init_depth=depth, init_width=2),
            "split.npz", str(tmp_path / name), {}, ("python", "-c", "pass"),
        )
        process = SimpleNamespace(pid=700 + depth, poll=lambda code=exit_code: code)
        runtime = JobRuntime(
            spec, status=RunStatus.RUNNING,
            attempt=1, device_mode=device, pid=process.pid, process=process,
        )
        runtimes[name] = runtime

    paused_identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=20,
        sample=20, suite_repeat=1, model_route="paused-ready", seed=4,
        split_sha256="split", run_signature="signature",
    )
    paused_spec = RunSpec(
        paused_identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All", init_depth=2, init_width=2), "split.npz",
        str(tmp_path / "paused-ready"), {}, ("python", "-c", "pass"),
    )
    paused_ready = JobRuntime(
        paused_spec, status=RunStatus.PAUSED, device_mode="gpu",
        pause_reason="gpu_memory_pressure",
    )

    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []},
        (*[runtime.spec for runtime in runtimes.values()], paused_spec),
    )
    memory_by_pid = {
        runtimes["pressure-peer"].pid: ProcessMemorySample(host_rss_mib=600, gpu_vram_mib=300),
    }
    provider = SimpleNamespace(
        sample_child=lambda pid, **_kwargs: memory_by_pid.get(pid, ProcessMemorySample(host_rss_mib=100, gpu_vram_mib=100)),
    )
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=4, max_retries=1, batch_backoff_factor=0.5,
            run_hardness_screen=False, post_launch_sample_delay_seconds=0,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=provider,
    )
    active = [runtimes[name] for name in ("failed-oom", "oom-peer", "pressure-peer", "cpu-worker")]
    queue = deque([runtimes["waiting"]])
    host = _host(500)
    gpu = _gpu(92)
    snapshot = ResourceSnapshot(
        host, gpu, 4, 3, 1, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), True, True, 4,
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: (active, [paused_ready], queue))
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    events = []
    monkeypatch.setattr(coordinator, "_resume_ready", lambda *_args: True)
    monkeypatch.setattr(
        coordinator, "_classify_failure",
        lambda *_args: (events.append("failure_classify") or "cuda_oom"),
    )
    monkeypatch.setattr(
        scheduler_module, "terminate_process_tree",
        lambda process: events.append(f"terminate:{process.pid}"),
    )

    def launch(runtime, *, device, launch_context, **_kwargs):
        if launch_context.get("resumed_from_pause"):
            events.append("resume_launch")
            runtime.process = SimpleNamespace(pid=990, poll=lambda: None)
            runtime.pid = runtime.process.pid
            runtime.device_mode = device
            runtime.status = RunStatus.RUNNING
            runtime.attempt += 1
            return runtime.process
        events.append("queue_launch")
        raise AssertionError("hard pressure and unmet lane thresholds must block admission")

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: setattr(coordinator, "shutdown_signal", int(signal.SIGTERM))

    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert "resume_launch" in events and "queue_launch" not in events
    assert events.index("resume_launch") < events.index("failure_classify") < next(
        index for index, value in enumerate(events) if value.startswith("terminate:")
    )
    assert list(queue) == [runtimes["waiting"], runtimes["failed-oom"]]
    assert runtimes["failed-oom"].status == RunStatus.RETRYING
    assert runtimes["failed-oom"].retry_count == 1
    assert runtimes["oom-peer"].status == RunStatus.PAUSED
    assert runtimes["oom-peer"].pause_reason == "peer_cuda_oom"
    assert runtimes["pressure-peer"].status == RunStatus.PAUSED
    assert runtimes["pressure-peer"].pause_reason == "gpu_memory_pressure+host_ram_pressure"
    assert paused_ready.attempt == 1 and paused_ready.last_error == "interrupted:SIGTERM"
    assert coordinator.counters.failed_attempts == coordinator.counters.retries == 1
    assert coordinator.admission.batch_scale == 0.5
    assert coordinator.admission.last_backoff_reason is None  # subsequent hard pause clears it


@pytest.mark.parametrize("exit_code", [0, 7, 9])
@pytest.mark.parametrize("host_pressure,gpu_pressure", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_full_pass_exit_retry_and_pressure_cross_matrix(
    tmp_path, monkeypatch, exit_code, host_pressure, gpu_pressure,
):
    """Cross terminal exit class with pressure domains in one scheduler pass."""
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample, ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    def make_runtime(name, depth, device, code=None):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0,
            fold=depth, sample=depth, suite_repeat=1, model_route=name, seed=4,
            split_sha256="split", run_signature="signature",
        )
        spec = RunSpec(
            identity, DatasetTaskRef("toy", "suite", "1"), "probe",
            ModelRoute("ADP-W-All", init_depth=depth, init_width=2),
            "split.npz", str(tmp_path / name), {}, ("python", "-c", "pass"),
        )
        process = SimpleNamespace(pid=700 + depth, poll=lambda result=code: result)
        runtime = JobRuntime(
            spec, status=RunStatus.RUNNING, attempt=1, device_mode=device,
            pid=process.pid, process=process,
        )
        return runtime

    failed = make_runtime("exiting-gpu", 1, "gpu", exit_code)
    gpu_peer = make_runtime("gpu-peer", 5, "gpu")
    cpu_peer = make_runtime("cpu-peer", 3, "cpu")
    runtimes = [failed, gpu_peer, cpu_peer]
    all_runtimes = tuple(runtimes)
    queue = deque()
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []},
        tuple(runtime.spec for runtime in runtimes),
    )
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=4, max_active_jobs=1, max_retries=1,
            batch_backoff_factor=0.5, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
        resource_provider=SimpleNamespace(sample_child=lambda pid, **_kwargs: {
            gpu_peer.pid: ProcessMemorySample(host_rss_mib=100, gpu_vram_mib=900),
            cpu_peer.pid: ProcessMemorySample(host_rss_mib=900, gpu_vram_mib=0),
        }.get(pid, ProcessMemorySample())),
    )
    snapshot = ResourceSnapshot(
        _host(500 if host_pressure else 4096),
        _gpu(50 if gpu_pressure else 600),
        3, 2, 1, 1000, 900, 900, 0, 0,
        frozenset(), frozenset(), host_pressure, gpu_pressure, 1,
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: (runtimes, [], queue))
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        coordinator, "_classify_failure",
        lambda _runtime, code: "cuda_oom" if code == 9 else f"nonzero_exit:{code}",
    )
    monkeypatch.setattr(
        scheduler_module, "terminate_process_tree",
        lambda _process: None,
    )
    coordinator.sleep = lambda _seconds: setattr(
        coordinator, "shutdown_signal", int(signal.SIGTERM),
    )

    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    failed_record = coordinator.state_store.load(failed.job_id)
    expected_retry = exit_code != 0
    assert failed.retry_count == int(expected_retry)
    assert failed.status == (RunStatus.RETRYING if expected_retry else RunStatus.COMPLETED)
    assert failed_record["last_exit_code"] == exit_code
    assert coordinator.counters.failed_attempts == int(expected_retry)
    assert coordinator.counters.retries == int(expected_retry)
    assert list(queue) == ([failed] if expected_retry else [])
    expected_paused = set()
    if exit_code == 9:
        expected_paused.add("gpu-peer")  # OOM peer pause happens during reap.
    if host_pressure and not gpu_pressure:
        expected_paused.add("cpu-peer")
    elif gpu_pressure and exit_code != 9:
        expected_paused.add("gpu-peer")
    assert {
        runtime.spec.identity.model_route for runtime in all_runtimes
        if runtime.status == RunStatus.PAUSED
    } == expected_paused
    # Combined pressure uses the GPU victim set. If OOM already paused that
    # peer, the hard-pressure pass finds no second eligible GPU victim.
    assert len([runtime for runtime in runtimes if runtime.status == RunStatus.PAUSED]) <= 2
    pause_events = list(csv.DictReader(
        coordinator.state_store.events_path.open(encoding="utf-8"),
    ))
    assert {event["job_id"] for event in pause_events if event["event"] == "pause"} == {
        runtime.job_id for runtime in all_runtimes
        if runtime.spec.identity.model_route in expected_paused
    }
    assert coordinator.admission.batch_scale == (0.5 if exit_code == 9 else 1.0)


@pytest.mark.parametrize(
    ("exit_kind", "exit_device"),
    [
        ("success", "cpu"), ("success", "gpu"),
        ("nonzero", "cpu"), ("nonzero", "gpu"),
        ("cuda_oom", "gpu"),
        ("exact_resume", "cpu"), ("exact_resume", "gpu"),
    ],
)
@pytest.mark.parametrize("host_pressure,gpu_pressure", [
    (False, False), (True, False), (False, True), (True, True),
])
@pytest.mark.parametrize("peer_devices", [(), ("cpu",), ("gpu",), ("cpu", "gpu")])
@pytest.mark.parametrize("max_active_jobs", [0, 1, 2])
@pytest.mark.parametrize("max_active_gpu_jobs", [0, 1, 2])
@pytest.mark.parametrize("gpu_gate_closed,host_gate_closed", [
    (False, False), (True, False), (False, True), (True, True),
])
@pytest.mark.parametrize("gpu_hold_active,cpu_hold_active", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_full_pass_exit_pressure_and_peer_lane_cross_product(
    _shared_exit_matrix_root, monkeypatch, exit_kind, exit_device, host_pressure, gpu_pressure,
    peer_devices, max_active_jobs, max_active_gpu_jobs, gpu_gate_closed,
    host_gate_closed, gpu_hold_active, cpu_hold_active,
):
    """Cross exits, pressure, lanes, caps, gates, and holds through a full pass."""
    import signal
    import shutil
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample, ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    tmp_path = _shared_exit_matrix_root
    # Reuse one isolated directory across this large parametrized matrix. A
    # unique pytest tmp_path for every case exhausts /tmp inodes long before
    # the matrix completes, while stale state is unnecessary between cases.
    shutil.rmtree(tmp_path / "run", ignore_errors=True)
    shutil.rmtree(tmp_path / "results", ignore_errors=True)

    exit_code = {"success": 0, "nonzero": 7, "cuda_oom": 9, "exact_resume": 11}[exit_kind]

    def make_runtime(name, depth, device, code=None):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0,
            fold=depth, sample=depth, suite_repeat=1, model_route=name, seed=4,
            split_sha256="split", run_signature="signature",
        )
        spec = RunSpec(
            identity, DatasetTaskRef("toy", "suite", "1"), "probe",
            ModelRoute("ADP-W-All", init_depth=depth, init_width=2),
            "split.npz", str(tmp_path / name), {}, ("python", "-c", "pass"),
        )
        process = SimpleNamespace(pid=800 + depth, poll=lambda result=code: result)
        return JobRuntime(
            spec, status=RunStatus.RUNNING, attempt=1, device_mode=device,
            pid=process.pid, process=process,
        )

    exiting = make_runtime(f"exiting-{exit_device}", 1, exit_device, exit_code)
    peers = [make_runtime(f"peer-{device}", index + 2, device)
             for index, device in enumerate(peer_devices)]
    waiting = make_runtime("waiting", 8, "cpu")
    waiting.status = RunStatus.QUEUED
    waiting.process = None
    waiting.pid = None
    runtimes = [exiting, *peers]
    all_runtimes = tuple(runtimes)
    queue = deque([waiting])
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []},
        tuple(runtime.spec for runtime in (*runtimes, waiting)),
    )
    rss_by_pid = {
        runtime.pid: ProcessMemorySample(
            host_rss_mib=900 if runtime.device_mode == "cpu" else 100,
            gpu_vram_mib=900 if runtime.device_mode == "gpu" else 0,
        )
        for runtime in peers
    }
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=4, max_active_jobs=max_active_jobs,
            max_active_gpu_jobs=max_active_gpu_jobs, max_retries=1,
            batch_backoff_factor=0.5, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
        wall_clock=lambda: 100.0,
        resource_provider=SimpleNamespace(
            sample_child=lambda pid, **_kwargs: rss_by_pid.get(pid, ProcessMemorySample()),
        ),
    )
    snapshot = ResourceSnapshot(
        _host(500 if host_pressure else 4096),
        _gpu(50 if gpu_pressure else 600),
        len(runtimes), sum(runtime.is_gpu for runtime in runtimes),
        sum(runtime.device_mode == "cpu" for runtime in runtimes),
        1000, 900, 900, 0, 0,
        frozenset(), frozenset(), host_pressure, gpu_pressure,
        min(4, max_active_jobs if max_active_jobs > 0 else 4, len(runtimes) + 1),
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: (runtimes, [], queue))
    def sample_snapshot(*_args):
        coordinator.admission.gpu_gate_closed = gpu_gate_closed
        coordinator.admission.host_gate_closed = host_gate_closed
        coordinator.admission.gpu_launches_enabled = not gpu_gate_closed
        coordinator.admission.cpu_launches_enabled = not host_gate_closed
        coordinator.admission.gpu_hold_until = 101.0 if gpu_hold_active else 0.0
        coordinator.admission.cpu_hold_until = 101.0 if cpu_hold_active else 0.0
        return snapshot, None, None

    monkeypatch.setattr(coordinator, "_sample_snapshot", sample_snapshot)
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(coordinator, "_resume_ready", lambda *_args, **_kwargs: False)

    def classify(_runtime, code):
        if exit_kind == "cuda_oom":
            return "cuda_oom"
        if exit_kind == "exact_resume":
            return "inexact_checkpoint_resume"
        return f"nonzero_exit:{code}"

    monkeypatch.setattr(coordinator, "_classify_failure", classify)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launched_devices = []

    def launch(runtime, *, device, **_kwargs):
        runtime.attempt += 1
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        runtime.pid = 9999
        runtime.process = SimpleNamespace(pid=9999, poll=lambda: None)
        launched_devices.append(device)
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: setattr(
        coordinator, "shutdown_signal", int(signal.SIGTERM),
    )

    try:
        coordinator.run()
    except SystemExit as stopped:
        assert stopped.code == 130

    retried = exit_kind in {"nonzero", "cuda_oom"}
    assert exiting.status == (
        RunStatus.RETRYING if retried else
        RunStatus.COMPLETED if exit_kind == "success" else RunStatus.FAILED
    )
    assert exiting.retry_count == int(retried)
    assert exiting.last_exit_code == exit_code
    assert coordinator.counters.failed_attempts == int(exit_kind != "success")
    assert coordinator.counters.retries == int(retried)

    expected_paused = set()
    gpu_peer = next((runtime for runtime in peers if runtime.is_gpu), None)
    cpu_peer = next((runtime for runtime in peers if runtime.device_mode == "cpu"), None)
    if exit_kind == "cuda_oom" and exit_device == "gpu" and gpu_peer is not None:
        expected_paused.add(gpu_peer.job_id)  # peer OOM uses architecture priority
    if host_pressure and not gpu_pressure:
        host_victim = cpu_peer or gpu_peer
        if host_victim is not None:
            expected_paused.add(host_victim.job_id)
    elif gpu_pressure and gpu_peer is not None and not (
        exit_kind == "cuda_oom" and exit_device == "gpu"
    ):
        expected_paused.add(gpu_peer.job_id)
    assert {runtime.job_id for runtime in all_runtimes if runtime.status == RunStatus.PAUSED} == expected_paused
    assert sum(runtime.status == RunStatus.PAUSED for runtime in all_runtimes) <= 2

    live_peers = [runtime for runtime in peers if runtime.job_id not in expected_paused]
    admission_capacity = min(
        4, max_active_jobs if max_active_jobs > 0 else 4, len(runtimes) + 1,
    )
    total_cap_full = len(live_peers) >= admission_capacity
    gpu_count = sum(runtime.is_gpu for runtime in live_peers)
    gpu_cap_full = max_active_gpu_jobs > 0 and gpu_count >= max_active_gpu_jobs
    # Every exit releases the host gate; a GPU exit releases both gates, while
    # a CPU exit leaves the independently closed GPU gate in place.
    remaining_gpu_gate = gpu_gate_closed and exit_device == "cpu"
    gpu_hold_blocks = gpu_hold_active
    cpu_hold_blocks = cpu_hold_active
    if gpu_hold_blocks and cpu_hold_blocks:
        expected_device = None
    else:
        gpu_ready = (
            not host_pressure and not gpu_pressure
            and not remaining_gpu_gate and not total_cap_full and not gpu_cap_full
            and not gpu_hold_blocks
        )
        cpu_ready = not host_pressure and not total_cap_full and not cpu_hold_blocks
        expected_device = "gpu" if gpu_ready else "cpu" if cpu_ready else None
    assert launched_devices == ([] if expected_device is None else [expected_device])
    expected_queue = (
        ([exiting] if retried else []) if expected_device is not None
        else ([waiting, exiting] if retried else [waiting])
    )
    assert list(queue) == expected_queue
    assert coordinator.admission.batch_scale == (
        0.5 if exit_kind == "cuda_oom" and expected_device is None else 1.0
    )


def test_shutdown_persists_active_as_queued_and_paused_as_paused(tmp_path, monkeypatch):
    import io
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.state import JobRuntime, RunStatus

    def make_runtime(label, status):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=label, seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        spec = RunSpec(
            identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
            str(tmp_path / label), {}, ("python", "-c", "pass"),
        )
        runtime = JobRuntime(spec, status=status, device_mode="cpu", pid=100 + len(label))
        runtime.process = SimpleNamespace(pid=runtime.pid, poll=lambda: None)
        runtime.log_handle = io.BytesIO()
        return runtime

    active = make_runtime("active", RunStatus.RUNNING)
    paused = make_runtime("paused", RunStatus.PAUSED)
    paused.pause_reason = "host_ram_pressure"
    plan = SimpleNamespace(run_signature="signature", runs=(active.spec, paused.spec))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    coordinator.shutdown_signal = int(signal.SIGTERM)
    expected_pids = [active.pid, paused.pid]
    terminated = []
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda process: terminated.append(process.pid))
    coordinator._shutdown([active], [paused])

    assert terminated == expected_pids
    assert active.status == RunStatus.QUEUED and active.pid is None and active.exit_code is None
    assert active.pause_reason == ""
    assert paused.status == RunStatus.PAUSED and paused.pause_reason == "host_ram_pressure"
    assert paused.pid is None and paused.exit_code is None
    assert paused.last_exit_code is None and paused.ended_at is None
    assert paused.last_error == "interrupted:SIGTERM"
    assert coordinator.state_store.load(active.job_id)["status"] == "queued"
    assert coordinator.state_store.load(active.job_id)["pause_reason"] == ""
    paused_record = coordinator.state_store.load(paused.job_id)
    assert paused_record["status"] == "paused"
    assert paused_record["pause_reason"] == "host_ram_pressure"
    assert paused_record["last_error"] == "interrupted:SIGTERM"
    assert paused_record["ended_at"] is None and paused_record["last_exit_code"] is None


def test_shutdown_job_state_write_failure_propagates_after_terminating_child(tmp_path, monkeypatch):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.state import JobRuntime, RunStatus

    spec = RunSpec(
        RunIdentity("toy", "suite", "shutdown-write", 0, 0, 0, 1, "shutdown-write", 4, "split", "signature"),
        DatasetTaskRef("toy", "suite", "shutdown-write"), "probe", ModelRoute("ADP-W-All"),
        "split.npz", str(tmp_path / "result"), {}, ("python", "worker.py"),
    )
    process = SimpleNamespace(pid=7654, poll=lambda: None)
    runtime = JobRuntime(
        spec, status=RunStatus.RUNNING, attempt=1, device_mode="cpu", pid=7654,
        process=process,
    )
    coordinator = SuiteCoordinator(
        SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,)),
        RunnerSettings(run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    coordinator.shutdown_signal = int(signal.SIGTERM)
    terminated = []
    saved_controller_state = []
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda child: terminated.append(child.pid))
    monkeypatch.setattr(
        coordinator.state_store, "save",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("state disk failure")),
    )
    monkeypatch.setattr(coordinator, "_save_controller_state", lambda: saved_controller_state.append(True))

    with pytest.raises(OSError, match="state disk failure"):
        coordinator._shutdown([runtime], [])

    assert terminated == [7654]
    assert runtime.process is None and runtime.pid is None
    assert runtime.status == RunStatus.QUEUED
    assert runtime.last_error == "interrupted:SIGTERM"
    assert saved_controller_state == []


def test_shutdown_swallows_controller_state_write_failure(tmp_path, monkeypatch):
    import signal

    coordinator = SuiteCoordinator(
        SuitePlan("signature", {"run_signature": "signature", "jobs": []}, ()),
        RunnerSettings(run_hardness_screen=False), run_root=tmp_path / "run",
        results_dir=tmp_path / "results", repository_root=tmp_path,
    )
    coordinator.shutdown_signal = int(signal.SIGTERM)

    def fail_controller_write():
        raise OSError("controller state disk failure")

    monkeypatch.setattr(coordinator, "_save_controller_state", fail_controller_write)
    coordinator._shutdown([], [])


def test_unexpected_controller_exception_does_not_run_signal_shutdown_cleanup(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="exception-path", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"), {},
        ("python", "-c", "pass"),
    )
    child = SimpleNamespace(pid=808, poll=lambda: None)
    active = [JobRuntime(
        spec, status=RunStatus.RUNNING, attempt=1, device_mode="cpu",
        pid=child.pid, process=child,
    )]

    def fail_host_sample():
        raise RuntimeError("unexpected synthetic sensor failure")

    coordinator = SuiteCoordinator(
        SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,)),
        RunnerSettings(cpu_only=True, run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, host_sampler=fail_host_sample,
    )
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: (active, [], deque()))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    terminations = []
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda process: terminations.append(process.pid))

    with pytest.raises(RuntimeError, match="unexpected synthetic sensor failure"):
        coordinator.run()

    # The ledger records no catch/finally for arbitrary controller exceptions;
    # signal and KeyboardInterrupt paths use cleanup, but this path does not.
    assert terminations == []
    assert active[0].process is child and active[0].status == RunStatus.RUNNING


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"])
def test_coordinator_installs_dispatches_and_restores_each_linux_shutdown_signal(
    tmp_path, monkeypatch, signal_name,
):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    sig = getattr(signal, signal_name)
    spec = RunSpec(
        RunIdentity("toy", "suite", "signal", 0, 0, 0, 1, "signal", 5, "split", "signature"),
        DatasetTaskRef("toy", "suite", "signal"), "probe", ModelRoute("ADP-W-All"),
        "split.npz", str(tmp_path / "result"), {}, ("python", "worker.py"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(run_hardness_screen=False, pressure_poll_seconds=0.1),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    runtime = JobRuntime(
        spec, status=RunStatus.RUNNING, attempt=1, device_mode="cpu", pid=777,
        process=SimpleNamespace(pid=777, poll=lambda: None),
    )
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([runtime], [], deque()))
    snapshot = ResourceSnapshot(
        _host(4096), None, 1, 0, 1, 0.0, 0.0, 0.0, 0.0, 0.0,
        frozenset(), frozenset(), False, False, 1,
    )
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
    previous = {item: ("previous-handler", int(item)) for item in signals}
    installed = {}
    restored = {}
    monkeypatch.setattr(scheduler_module.signal, "getsignal", lambda item: previous[item])

    def set_handler(item, handler):
        if callable(handler):
            installed[item] = handler
        else:
            restored[item] = handler

    monkeypatch.setattr(scheduler_module.signal, "signal", set_handler)

    immediate_exit_codes = []

    def trigger(_seconds):
        with pytest.raises(SystemExit) as immediate:
            installed[sig](int(sig), None)
        immediate_exit_codes.append(immediate.value.code)
        raise immediate.value

    coordinator.sleep = trigger
    terminated = []
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda child: terminated.append(child.pid))
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert immediate_exit_codes == [130]
    assert coordinator.shutdown_signal == int(sig)
    assert set(installed) == set(signals)
    assert restored == previous
    assert terminated == [777]
    assert runtime.status.value == "queued" and runtime.pid is None


def test_coordinator_keyboard_interrupt_restores_all_linux_signal_handlers(tmp_path, monkeypatch):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot

    spec = RunSpec(
        RunIdentity("toy", "suite", "keyboard", 0, 0, 0, 1, "keyboard", 5, "split", "signature"),
        DatasetTaskRef("toy", "suite", "keyboard"), "probe", ModelRoute("ADP-W-All"),
        "split.npz", str(tmp_path / "result"), {}, ("python", "worker.py"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))
    coordinator = SuiteCoordinator(
        plan, RunnerSettings(run_hardness_screen=False, pressure_poll_seconds=0.1),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(
        _host(0), None, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        frozenset(), frozenset(), True, False, 1,
    )
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
    previous = {item: ("previous-handler", int(item)) for item in signals}
    installed = {}
    restored = {}
    monkeypatch.setattr(scheduler_module.signal, "getsignal", lambda item: previous[item])

    def set_handler(item, handler):
        if callable(handler):
            installed[item] = handler
        else:
            restored[item] = handler

    monkeypatch.setattr(scheduler_module.signal, "signal", set_handler)

    def interrupt(_seconds):
        raise KeyboardInterrupt

    coordinator.sleep = interrupt
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert coordinator.shutdown_signal is None
    assert set(installed) == set(signals)
    assert restored == previous


def test_launcher_cli_matches_source_keyboard_interrupt_notice(monkeypatch, capsys):
    import simple_adp.runner.launcher as launcher

    def interrupted(_argv=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(launcher, "main", interrupted)
    assert launcher.cli_main([]) == 130
    assert capsys.readouterr().out == (
        "\n[INTERRUPT] Caught KeyboardInterrupt. Launcher-owned jobs were cleaned up and marked resumable.\n"
    )


def test_full_controller_pass_combines_lane_cap_and_post_launch_holds(tmp_path, monkeypatch):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import RunStatus

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    specs = []
    for index in range(3):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=index,
            sample=0, suite_repeat=1, model_route=f"route-{index}", seed=4,
            split_sha256="split", run_signature="signature",
        )
        ref = DatasetTaskRef("toy", "suite", "1")
        specs.append(RunSpec(
            identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
            str(tmp_path / f"result-{index}"), {}, ("python", "-c", "pass"),
        ))
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(specs))
    settings = RunnerSettings(
        concurrency=3, max_active_gpu_jobs=1,
        post_launch_sample_delay_seconds=5, pressure_poll_seconds=0.1,
    )
    coordinator = SuiteCoordinator(
        plan, settings, run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(
        host=_host(8192), gpu=_gpu(8192), active_count=0, active_gpu_count=0,
        active_cpu_count=0, active_rss_mib=0, latest_child_rss_mib=0,
        latest_child_vram_mib=0, external_host_mib=0, external_gpu_mib=0,
        external_host_pids=frozenset(), external_gpu_pids=frozenset(),
        host_pressure=False, gpu_pressure=False, launch_capacity=3,
    )
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    launches = []

    class LiveProcess:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    def launch(runtime, *, device, **_kwargs):
        runtime.attempt += 1
        runtime.process = LiveProcess(100 + len(launches))
        runtime.pid = runtime.process.pid
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = device
        launches.append((runtime.job_id, device))
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    hold_events = []
    monkeypatch.setattr(
        scheduler_module.LOG, "info",
        lambda *args: hold_events.append(("log", args)),
    )

    def no_wait(_seconds):
        hold_events.append(("sleep", _seconds))
        if coordinator.counters.launched >= 2:
            coordinator.shutdown_signal = int(signal.SIGTERM)

    coordinator.sleep = no_wait
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    # The GPU cap and active GPU hold make the second route fall back to CPU;
    # both lane holds then block the third route despite unused total capacity.
    assert [device for _job_id, device in launches] == ["gpu", "cpu"]
    assert len(launches) == 2
    logs = [event[1] for event in hold_events if event[0] == "log"]
    launch_event_logs = [args for args in logs if args[0].startswith("[%s]") and args[1] == "LAUNCH"]
    hold_logs = [args for args in logs if args[0].startswith("[HOLD]")]
    resource_logs = [args for args in logs if args[0].startswith("[RESOURCE]")]
    assert hold_logs == [
        (
            "[HOLD] launched=%s device=%s waiting=%.1fs before next resource poll",
            job_id,
            device,
            5,
        )
        for job_id, device in launches
    ]
    assert [args[0] for args in resource_logs] == [
        "[RESOURCE] launch=%s queue=%d active=%d cap=%d host_avail=%d host_rss=%d gpu_free=%s gpu_used=%s last_child_rss=%s last_child_vram=%s gpu_hold=%.1fs cpu_hold=%.1fs"
    ] * 2
    assert [args[1:5] for args in resource_logs] == [
        (job_id, 3 - index, index, 3)
        for index, (job_id, _device) in enumerate(launches)
    ]
    assert [args[1:] for args in launch_event_logs] == [
        (
            "LAUNCH", job_id, "1", 1, "model", "ADP-W-All", 1, 1, 1,
            device, device, 0,
        )
        for job_id, device in launches
    ]
    for index, event in enumerate(hold_events):
        if event[0] == "log" and event[1][0].startswith("[HOLD]"):
            assert hold_events[index - 1][0] == "log"
            assert hold_events[index - 1][1][0].startswith("[%s]")
            assert hold_events[index - 1][1][1] == "LAUNCH"
            assert hold_events[index - 2][0] == "log"
            assert hold_events[index - 2][1][0].startswith("[RESOURCE]")
            assert hold_events[index + 1] == ("sleep", 5)


def test_full_controller_cpu_hold_does_not_block_gpu_lane(tmp_path, monkeypatch):
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import JobRuntime, RunStatus

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    specs = []
    for index in range(2):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=index,
            sample=0, suite_repeat=1, model_route=f"cpu-hold-{index}", seed=4,
            split_sha256="split", run_signature="signature",
        )
        specs.append(RunSpec(
            identity, DatasetTaskRef("toy", "suite", "1"), "probe",
            ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / f"result-{index}"),
            {}, ("python", "-c", "pass"),
        ))
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(specs))
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=2, post_launch_sample_delay_seconds=5,
            pressure_poll_seconds=0.1, run_hardness_screen=False,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, wall_clock=lambda: 10.0,
    )
    snapshots = [
        ResourceSnapshot(
            _host(8192), None, 0, 0, 0, 0, 0, 0, 0, 0,
            frozenset(), frozenset(), False, False, 2,
        ),
        ResourceSnapshot(
            _host(8192), _gpu(8192), 1, 0, 1, 0, 0, 0, 0, 0,
            frozenset(), frozenset(), False, False, 2,
        ),
    ]
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshots.pop(0), None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_module, "terminate_process_tree", lambda _process: None)
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([], [], deque(
        JobRuntime(spec) for spec in specs
    )))
    launches = []

    class LiveProcess:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    def launch(runtime, *, device, **_kwargs):
        runtime.process = LiveProcess(200 + len(launches))
        runtime.pid = runtime.process.pid
        runtime.status = RunStatus.RUNNING
        runtime.device_mode = device
        launches.append((runtime.job_id, device))
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)

    def stop_after_two_launches(_seconds):
        if len(launches) == 2:
            coordinator.shutdown_signal = int(signal.SIGTERM)

    coordinator.sleep = stop_after_two_launches
    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert [device for _job_id, device in launches] == ["cpu", "gpu"]
    assert coordinator.admission.cpu_hold_until == 15.0
    assert coordinator.admission.gpu_hold_until == 15.0


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("pid_disappears", [False, True])
def test_full_pass_external_drop_at_threshold_reopens_gate_and_resumes_at_lane_limit(
    tmp_path, monkeypatch, device, pid_disappears,
):
    """Pin external-drop equality, PID disappearance, and lane resume equality together."""
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route=f"threshold-{device}", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"),
        {}, ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))

    class SyntheticResources:
        def sample_host(self):
            return _host(1280 if device == "cpu" else 4096)

        def sample_gpu(self, _device_index):
            return _gpu(500) if device == "gpu" else None

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, _pid, **_kwargs):
            return ProcessMemorySample()

        def sample_external_host(self, _excluded):
            if device == "cpu":
                pids = frozenset() if pid_disappears else frozenset({90})
                return ProcessMemorySample(host_rss_mib=500, pid_set=pids)
            return ProcessMemorySample()

        def sample_external_gpu(self, _device_index, _excluded):
            if device == "gpu":
                pids = frozenset() if pid_disappears else frozenset({91})
                return ProcessMemorySample(gpu_vram_mib=500, pid_set=pids)
            return ProcessMemorySample()

    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            external_memory_drop_mib=500,
            host_ram_resume_avail_mib=1280,
            gpu_resume_avail_mib=500,
            run_hardness_screen=False,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=SyntheticResources(),
    )
    coordinator.memory.values.update({
        "external_host_mib": 1000 if device == "cpu" else 0,
        "external_host_pids": [90] if device == "cpu" else [],
        "external_gpu_mib": 1000 if device == "gpu" else 0,
        "external_gpu_pids": [91] if device == "gpu" else [],
    })
    if device == "cpu":
        coordinator.admission.host_gate_closed = True
        coordinator.admission.cpu_launches_enabled = False
        reason = "host_ram_pressure"
    else:
        coordinator.admission.gpu_gate_closed = True
        coordinator.admission.gpu_gate_lock_job_id = "gpu-gate-owner"
        coordinator.admission.gpu_launches_enabled = False
        reason = "gpu_memory_pressure"
    paused = JobRuntime(
        spec, status=RunStatus.PAUSED, device_mode=device, pause_reason=reason,
    )
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(coordinator, "_build_runtimes", lambda: ([], [paused], deque()))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    class Exited:
        pid = 909

        def poll(self):
            return 0

    launched = []

    def launch(runtime, *, device, launch_context, **_kwargs):
        gate_closed = (
            coordinator.admission.host_gate_closed if device == "cpu"
            else coordinator.admission.gpu_gate_closed
        )
        launched.append((device, launch_context, gate_closed))
        runtime.process = Exited()
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        runtime.attempt += 1
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    summary = coordinator.run()

    external_domain = "host" if device == "cpu" else "gpu"
    assert coordinator.last_resource_snapshot[f"external_{external_domain}_reopened"] is pid_disappears
    assert coordinator.last_resource_snapshot[f"external_{external_domain}_drop_mib"] == pytest.approx(500)
    assert len(launched) == 1
    assert launched[0][0] == device
    assert launched[0][1]["resumed_from_pause"] is True
    # Resume readiness follows the matching resource threshold even when the
    # external-drop gate stays closed because its tracked PID is still alive.
    assert launched[0][2] is not pid_disappears
    assert paused.status == RunStatus.COMPLETED
    assert paused.attempt == 1
    assert summary["completed"] == 1


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("pid_disappears", [False, True])
@pytest.mark.parametrize("resume_exit_code", [0, 9])
def test_full_pass_external_reopen_resumes_through_holds_but_blocks_queued_launch(
    tmp_path, monkeypatch, device, pid_disappears, resume_exit_code,
):
    """Cross PID-drop, resume exit, holds, queued admission, and shutdown."""
    import signal
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    def make_spec(route):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=route, seed=4,
            split_sha256="split", run_signature="signature",
        )
        return RunSpec(
            identity, DatasetTaskRef("toy", "suite", "1"), "probe",
            ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / route),
            {}, ("python", "-c", "pass"),
        )

    paused_spec, queued_spec = make_spec(f"paused-{device}"), make_spec(f"queued-{device}")
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []},
        (paused_spec, queued_spec),
    )

    class SyntheticResources:
        def sample_host(self):
            return _host(1280)

        def sample_gpu(self, _device_index):
            return _gpu(500) if device == "gpu" else None

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, _pid, **_kwargs):
            return ProcessMemorySample()

        def sample_external_host(self, _excluded):
            if device == "cpu":
                pids = frozenset() if pid_disappears else frozenset({90})
                return ProcessMemorySample(host_rss_mib=500, pid_set=pids)
            return ProcessMemorySample()

        def sample_external_gpu(self, _device_index, _excluded):
            if device == "gpu":
                pids = frozenset() if pid_disappears else frozenset({91})
                return ProcessMemorySample(gpu_vram_mib=500, pid_set=pids)
            return ProcessMemorySample()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("simple_adp.runner.admission.parent_cpu_masked", lambda: False)
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            cpu_only=device == "cpu", concurrency=1, external_memory_drop_mib=500,
            host_ram_resume_avail_mib=1280,
            gpu_resume_avail_mib=500,
            post_launch_sample_delay_seconds=30,
            run_hardness_screen=False,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=SyntheticResources(),
        wall_clock=lambda: 100.0,
    )
    external_domain = "host" if device == "cpu" else "gpu"
    coordinator.memory.values.update({
        "external_host_mib": 1000 if device == "cpu" else 0,
        "external_host_pids": [90] if device == "cpu" else [],
        "external_gpu_mib": 1000 if device == "gpu" else 0,
        "external_gpu_pids": [91] if device == "gpu" else [],
    })
    if device == "cpu":
        coordinator.admission.host_gate_closed = True
        coordinator.admission.host_gate_lock_job_id = "external-host-owner"
        coordinator.admission.cpu_launches_enabled = False
        pause_reason = "host_ram_pressure"
    else:
        coordinator.admission.gpu_gate_closed = True
        coordinator.admission.gpu_gate_lock_job_id = "external-gpu-owner"
        coordinator.admission.gpu_launches_enabled = False
        pause_reason = "gpu_memory_pressure"
    coordinator.admission.cpu_hold_until = 999.0
    coordinator.admission.gpu_hold_until = 999.0
    paused = JobRuntime(
        paused_spec, status=RunStatus.PAUSED, device_mode=device,
        pause_reason=pause_reason,
    )
    queued = JobRuntime(queued_spec)
    queue = deque([queued])
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(
        coordinator, "_build_runtimes",
        lambda: ([], [paused], queue),
    )
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    class Exited:
        pid = 919

        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

    launches = []

    def launch(runtime, *, device, launch_context, **_kwargs):
        gate_closed = (
            coordinator.admission.host_gate_closed if device == "cpu"
            else coordinator.admission.gpu_gate_closed
        )
        launches.append((runtime.job_id, device, launch_context, gate_closed))
        assert launch_context.get("resumed_from_pause") is True
        runtime.process = Exited(resume_exit_code)
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        runtime.attempt += 1
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: setattr(
        coordinator, "shutdown_signal", int(signal.SIGTERM),
    )

    with pytest.raises(SystemExit) as stopped:
        coordinator.run()

    assert stopped.value.code == 130
    assert coordinator.last_resource_snapshot[f"external_{external_domain}_reopened"] is pid_disappears
    assert len(launches) == 1
    assert launches[0][0:2] == (paused.job_id, device)
    if resume_exit_code == 0:
        # Completion can release a lane gate after this launch, so observe the
        # gate at resume time to distinguish external reopen from resume bypass.
        assert launches[0][3] is not pid_disappears
        assert paused.status == RunStatus.COMPLETED
        assert list(queue) == [queued]
    else:
        assert launches[0][3] is not pid_disappears
        assert paused.status == RunStatus.RETRYING
        assert paused.retry_count == 1
        assert list(queue) == [queued, paused]
        assert coordinator.counters.failed_attempts == coordinator.counters.retries == 1
    assert queued.status == RunStatus.QUEUED and queued.attempt == 0
    hold_until = coordinator.admission.cpu_hold_until if device == "cpu" else coordinator.admission.gpu_hold_until
    assert hold_until == 999.0


@pytest.mark.parametrize("host_pid_disappears", [False, True])
@pytest.mark.parametrize("gpu_pid_disappears", [False, True])
def test_full_pass_reopens_host_and_gpu_gates_independently_before_multi_lane_resume(
    tmp_path, monkeypatch, host_pid_disappears, gpu_pid_disappears,
):
    """Both external-drop domains are sampled together; each gate needs its own PID exit."""
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ProcessMemorySample
    from simple_adp.runner.state import JobRuntime, RunStatus

    def make_spec(route):
        identity = RunIdentity(
            provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
            sample=0, suite_repeat=1, model_route=route, seed=4,
            split_sha256="split", run_signature="signature",
        )
        return RunSpec(
            identity, DatasetTaskRef("toy", "suite", "1"), "probe",
            ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / route),
            {}, ("python", "-c", "pass"),
        )

    cpu_spec, gpu_spec = make_spec("paused-cpu"), make_spec("paused-gpu")
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": []},
        (cpu_spec, gpu_spec),
    )

    class SyntheticResources:
        def sample_host(self):
            return _host(1280)

        def sample_gpu(self, _device_index):
            return _gpu(500)

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, _pid, **_kwargs):
            return ProcessMemorySample()

        def sample_external_host(self, _excluded):
            pids = frozenset() if host_pid_disappears else frozenset({90})
            return ProcessMemorySample(host_rss_mib=500, pid_set=pids)

        def sample_external_gpu(self, _device_index, _excluded):
            pids = frozenset() if gpu_pid_disappears else frozenset({91})
            return ProcessMemorySample(gpu_vram_mib=500, pid_set=pids)

    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=1, max_active_jobs=1, max_active_gpu_jobs=1,
            external_memory_drop_mib=500, host_ram_resume_avail_mib=1280,
            gpu_resume_avail_mib=500, run_hardness_screen=False,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=SyntheticResources(),
    )
    coordinator.memory.values.update({
        "external_host_mib": 1000,
        "external_host_pids": [90],
        "external_gpu_mib": 1000,
        "external_gpu_pids": [91],
    })
    coordinator.admission.host_gate_closed = True
    coordinator.admission.gpu_gate_closed = True
    coordinator.admission.host_gate_lock_job_id = "host-owner"
    coordinator.admission.gpu_gate_lock_job_id = "gpu-owner"
    coordinator.admission.cpu_launches_enabled = False
    coordinator.admission.gpu_launches_enabled = False
    cpu_paused = JobRuntime(
        cpu_spec, status=RunStatus.PAUSED, device_mode="cpu",
        pause_reason="host_ram_pressure",
    )
    gpu_paused = JobRuntime(
        gpu_spec, status=RunStatus.PAUSED, device_mode="gpu",
        pause_reason="gpu_memory_pressure",
    )
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: None)
    monkeypatch.setattr(
        coordinator, "_build_runtimes",
        lambda: ([], [cpu_paused, gpu_paused], deque()),
    )
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    class Exited:
        pid = 929

        def poll(self):
            return 0

    launched = []

    def launch(runtime, *, device, launch_context, **_kwargs):
        launched.append((
            runtime.job_id,
            device,
            coordinator.admission.host_gate_closed,
            coordinator.admission.gpu_gate_closed,
            launch_context,
        ))
        runtime.process = Exited()
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        runtime.attempt += 1
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    summary = coordinator.run()

    assert coordinator.last_resource_snapshot["external_host_reopened"] is host_pid_disappears
    assert coordinator.last_resource_snapshot["external_gpu_reopened"] is gpu_pid_disappears
    assert len(launched) == 2  # resume bypasses active caps and reuses this pass's sample
    by_device = {device: row for row in launched for device in (row[1],)}
    assert by_device["cpu"][0] == cpu_paused.job_id
    assert by_device["gpu"][0] == gpu_paused.job_id
    for row in launched:
        assert row[4]["resumed_from_pause"] is True
        assert row[2] is not host_pid_disappears
        assert row[3] is not gpu_pid_disappears
    assert cpu_paused.status == gpu_paused.status == RunStatus.COMPLETED
    assert summary["completed"] == 2


def test_new_child_receives_resource_capped_launch_budget(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="route", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"), {},
        ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=10, cpu_only=True, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    snapshot = ResourceSnapshot(
        _host(8192), None, 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 2,
    )
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)
    observed = {}

    class FinishedProcess:
        pid = 321

        def poll(self):
            return 0

    def launch(runtime, *, launch_budget, device, **_kwargs):
        observed["launch_budget"] = launch_budget
        observed["device"] = device
        runtime.attempt += 1
        runtime.process = FinishedProcess()
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: None

    coordinator.run()

    # Scheduler concurrency is ten, but this sample only admits two slots.
    assert observed == {"launch_budget": 2, "device": "cpu"}


def test_launch_context_records_scaled_and_resource_capped_budget(tmp_path, monkeypatch):
    import simple_adp.runner.scheduler as scheduler_module
    from simple_adp.runner.resources import ResourceSnapshot
    from simple_adp.runner.state import RunStatus

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="scaled-budget", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="scaled-budget", seed=4,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "scaled-budget"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "result"), {},
        ("python", "-c", "pass"),
    )
    plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,))
    coordinator = SuiteCoordinator(
        plan,
        RunnerSettings(
            concurrency=10, cpu_only=True, run_hardness_screen=False,
            post_launch_sample_delay_seconds=0, pressure_poll_seconds=0.1,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    # OOM backoff scales ceil(10 * 0.25) to three; resource capacity is eight.
    snapshot = ResourceSnapshot(
        _host(8192), None, 0, 0, 0, 0, 0, 0, 0, 0,
        frozenset(), frozenset(), False, False, 8,
    )
    monkeypatch.setattr(coordinator, "_load_controller_state", lambda: setattr(coordinator.admission, "batch_scale", 0.25))
    monkeypatch.setattr(coordinator, "_sample_snapshot", lambda *_args: (snapshot, None, None))
    monkeypatch.setattr(coordinator, "_status", lambda *_args, **_kwargs: None)

    observed = {}

    class FinishedProcess:
        pid = 322

        def poll(self):
            return 0

    def launch(runtime, *, launch_budget, device, launch_context, **_kwargs):
        observed.update(
            launch_budget=launch_budget,
            device=device,
            context_budget=launch_context["launch_capacity"],
            batch_scale=launch_context["batch_scale"],
            resource_capacity=launch_context["resource_launch_capacity"],
            host_available=launch_context["host_available_effective_mib"],
            active_counts=(
                launch_context["active_count"],
                launch_context["active_gpu_count"],
                launch_context["active_cpu_count"],
            ),
            thresholds=(
                launch_context["host_ram_stop_avail_mib"],
                launch_context["host_ram_resume_avail_mib"],
                launch_context["gpu_stop_avail_mib"],
                launch_context["gpu_resume_avail_mib"],
            ),
        )
        runtime.attempt += 1
        runtime.process = FinishedProcess()
        runtime.pid = runtime.process.pid
        runtime.device_mode = device
        runtime.status = RunStatus.RUNNING
        return runtime.process

    monkeypatch.setattr(scheduler_module, "launch_job", launch)
    coordinator.sleep = lambda _seconds: None

    coordinator.run()

    assert observed == {
        "launch_budget": 3,
        "device": "cpu",
        "context_budget": 3,
        "batch_scale": 0.25,
        "resource_capacity": 8,
        "host_available": 8192,
        "active_counts": (0, 0, 0),
        "thresholds": (1024.0, 1280.0, 100.0, 500.0),
    }
