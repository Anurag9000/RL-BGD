"""Child launch, persisted-process recovery, and owned process-tree cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import psutil

from .admission import DeviceMode
from .artifacts import estimator_checkpoint_path
from .compat import local_timestamp
from .config import RunnerSettings
from .plan import RunSpec
from .resources import process_tree_pids
from .runtime_tuning import launcher_child_env
from .state import JobRuntime, JobStateStore, RunStatus, atomic_write_json


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...


def close_log_handle(handle: Any) -> None:
    """Flush and close a child log independently, swallowing either failure."""
    if handle is None:
        return
    try:
        handle.flush()
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


@dataclass(slots=True)
class RecoveredProcessHandle:
    pid: int

    def poll(self) -> int | None:
        if process_is_live(self.pid):
            return None
        # The true exit status is unavailable after reconnecting to a PID;
        # callers classify this synthetic value as an unknown-exit requeue.
        return 1


def process_is_live(pid: int | None) -> bool:
    try:
        pid_int = int(pid)
        if pid_int <= 0:
            return False
    except Exception:
        return False
    try:
        if not psutil.pid_exists(pid_int):
            return False
        process = psutil.Process(pid_int)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _normalize_argv(command: Any) -> tuple[str, ...]:
    if not command:
        return ()
    try:
        return tuple(str(part) for part in command if str(part))
    except Exception:
        return ()


def find_exact_argv(command: tuple[str, ...] | list[str]) -> psutil.Process | None:
    normalized = _normalize_argv(command)
    if not normalized:
        return None
    matches: list[tuple[float, int, psutil.Process]] = []
    try:
        for process in psutil.process_iter(attrs=["pid", "cmdline", "create_time"]):
            try:
                argv = _normalize_argv(process.info.get("cmdline"))
                if not argv or argv != normalized:
                    continue
                pid = int(process.info["pid"])
                create_time = float(process.info.get("create_time") or 0.0)
                if pid > 0 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    matches.append((create_time, pid, process))
            except Exception:
                continue
    except Exception:
        return None
    return min(matches, key=lambda item: (item[0], item[1]))[2] if matches else None


def terminate_process_tree(root: int | subprocess.Popen[Any], timeout_seconds: float = 10.0) -> None:
    """Terminate descendants then parent; wait, kill survivors, wait, kill once more."""
    process_handle = root if isinstance(root, subprocess.Popen) else None
    if process_handle is not None:
        if process_handle.poll() is not None:
            return
    pid = int(root.pid if process_handle is not None else root)
    try:
        parent = psutil.Process(pid)
    except Exception:
        if process_handle is not None:
            try:
                process_handle.terminate()
            except Exception:
                pass
        return
    descendants = parent.children(recursive=True)
    processes = [parent, *descendants]
    for child in reversed(descendants):
        try:
            child.terminate()
        except Exception:
            pass
    try:
        parent.terminate()
    except Exception:
        pass
    timeout = max(0.1, float(timeout_seconds))
    _gone, alive = psutil.wait_procs(processes, timeout=timeout)
    if not alive:
        if process_handle is not None:
            try:
                process_handle.wait(timeout=0.1)
            except Exception:
                pass
        return
    for child in alive:
        try:
            child.kill()
        except Exception:
            pass
    _gone, alive_again = psutil.wait_procs(alive, timeout=timeout)
    for child in alive_again:
        try:
            child.kill()
        except Exception:
            pass
    if process_handle is not None:
        try:
            process_handle.wait(timeout=timeout)
        except Exception:
            pass


def terminate_standalone_pressure_tree(
    process_handle: subprocess.Popen[Any], timeout_seconds: float = 10.0,
) -> None:
    """Match the standalone pressure wrapper's TERM/wait/KILL sequence.

    The standalone source always performs its second ``wait_procs`` call,
    including when the first wait reports no survivors, and uses the configured
    timeout for the final root wait. The central scheduler's helper has a
    different early-return path, so this lifecycle stays explicit.
    """
    try:
        parent = psutil.Process(int(process_handle.pid))
    except Exception:
        try:
            process_handle.terminate()
        except Exception:
            pass
        return

    descendants = parent.children(recursive=True)
    process_list = [parent, *descendants]
    for child in reversed(descendants):
        try:
            child.terminate()
        except Exception:
            pass
    try:
        parent.terminate()
    except Exception:
        pass

    timeout = max(0.1, float(timeout_seconds))
    _gone, alive = psutil.wait_procs(process_list, timeout=timeout)
    for child in alive:
        try:
            child.kill()
        except Exception:
            pass
    _gone, still_alive = psutil.wait_procs(alive, timeout=timeout)
    for child in still_alive:
        try:
            child.kill()
        except Exception:
            pass
    try:
        process_handle.wait(timeout=timeout)
    except Exception:
        pass


def terminate_process_tree_after_grace(root_pid: int, grace_seconds: float = 2.0) -> None:
    """Match the Linux outer-wrapper TERM, fixed grace, fresh-tree KILL sequence."""
    def signal_tree(sig: int) -> None:
        try:
            parent = psutil.Process(int(root_pid))
        except Exception:
            return

        def signal_postorder(process: psutil.Process) -> None:
            # `_watchdog_kill_process_tree` in the source shell wrapper runs
            # itself for each direct child before signaling that child's
            # parent. A reversed flat recursive listing is not equivalent for
            # nested trees (psutil lists descendants in traversal order).
            try:
                children = process.children(recursive=False)
            except Exception:
                children = []
            for child in children:
                signal_postorder(child)
            try:
                process.send_signal(sig)
            except Exception:
                pass

        signal_postorder(parent)

    signal_tree(signal.SIGTERM)
    time.sleep(max(0.0, float(grace_seconds)))
    # The shell reference performs a second recursive walk after the grace
    # period, so descendants created during that interval are included.
    signal_tree(signal.SIGKILL)


def launch_job(
    runtime: JobRuntime,
    *,
    settings: RunnerSettings,
    state_store: JobStateStore,
    repository_root: str | Path,
    launch_budget: int,
    device: DeviceMode,
    launch_context: dict[str, object] | None = None,
) -> subprocess.Popen[bytes]:
    runtime.attempt += 1
    runtime.device_mode = device
    runtime.started_at = time.time()
    runtime.ended_at = None
    runtime.exit_code = None
    runtime.last_exit_code = None
    runtime.status = RunStatus.LAUNCHING
    runtime.last_error = None
    runtime.launch_context = dict(launch_context or {})
    runtime.last_progress_ts = time.time()
    job_dir = state_store.job_dir(runtime)
    job_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(runtime.spec.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(job_dir / "command.json", list(runtime.spec.command))
    atomic_write_json(job_dir / "job_metadata.json", {
        **runtime.spec.as_dict(),
        "attempt": runtime.attempt,
        "device_mode": device,
        "launch_context": launch_context or {},
    })
    if launch_context:
        atomic_write_json(job_dir / "launch_context.json", launch_context)
    state_store.save(runtime)
    log_path = job_dir / "job.log"
    env = launcher_child_env(
        os.environ.copy(),
        job_key=runtime.job_id,
        launch_budget=launch_budget,
        shared_cpu=False,
        cpu_only=settings.cpu_only or device == "cpu",
        device_index=settings.gpu_device_index,
        device_mode=device,
    )
    checkpoint_mirrors = {
        "last": "checkpoint_last",
        "best": "checkpoint_best",
        "best_batch": "checkpoint_best_batch",
    }
    env.update({
        "SIMPLE_ADP_RESULTS_DIR": runtime.spec.result_dir,
        "SIMPLE_ADP_RUN_ROOT": str(state_store.run_root),
        "SIMPLE_ADP_JOB_ID": runtime.job_id,
        "SIMPLE_ADP_SEED": str(runtime.spec.identity.seed),
        "PYTHONHASHSEED": str(runtime.spec.identity.seed),
        "TABULAR_CHILD_SHARED_CPU": "0",
        "TABULAR_CPU_WORKERS": "0",
        "TABULAR_CPU_JOB_CONCURRENCY": "1" if launch_budget >= 10**9 else str(max(1, launch_budget)),
        "SIMPLE_ADP_CHECKPOINT_PATH": str(
            (launch_context or {}).get("checkpoint_path")
            or estimator_checkpoint_path(runtime.spec.result_dir)
        ),
        **{
            f"SIMPLE_ADP_CHECKPOINT_{key.upper()}_PATH": str(
                (launch_context or {}).get(context_key) or (job_dir / f"{context_key}.pt")
            )
            for key, context_key in checkpoint_mirrors.items()
        },
    })
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(
        f"\n=== LAUNCH {runtime.job_id} attempt={runtime.attempt} device={device} ===\n"
        + "COMMAND: " + " ".join(runtime.spec.command) + "\n"
        + ("LAUNCH_CONTEXT: " + json.dumps(launch_context, sort_keys=True) + "\n" if launch_context else "")
    )
    log_handle.flush()
    try:
        process = subprocess.Popen(
            list(runtime.spec.command),
            cwd=str(repository_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
            creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0),
        )
    except Exception:
        # The source launch helper directly closes after Popen failure; unlike
        # retry/finalization, it does not make a separate explicit flush call.
        try:
            log_handle.close()
        except Exception:
            pass
        raise
    runtime.process = process
    runtime.pid = process.pid
    runtime.log_handle = log_handle
    runtime.log_path = str(log_path)
    runtime.status = RunStatus.RUNNING
    runtime.recovered_process = False
    state_store.save(runtime)
    return process


def reconcile_runtime(
    runtime: JobRuntime,
    record: dict[str, Any] | None,
    *,
    state_store: JobStateStore | None = None,
) -> str:
    """Attach live process state or classify it for queue/suspended recovery."""
    if record is None:
        return "queued"
    raw_status = str(record.get("status", "queued")).lower()
    try:
        persisted_status = RunStatus(raw_status)
    except ValueError:
        persisted_status = RunStatus.QUEUED
    runtime.attempt = _safe_record_int(record.get("attempt", 0))
    runtime.retry_count = _safe_record_int(record.get("retry_count", 0))
    saved_pid = _safe_record_int(record.get("pid"))
    runtime.pid = saved_pid if saved_pid > 0 else None
    # Match load_existing_runtime's source defaults for legacy records that
    # predate these fields or persisted them as null/empty.
    runtime.device_mode = str(record.get("device_mode") or "gpu")
    runtime.pause_reason = str(record.get("pause_reason") or "")
    runtime.pause_count = _safe_record_int(record.get("pause_count", 0))
    runtime.gpu_pause_count = _safe_record_int(record.get("gpu_pause_count", 0))
    runtime.host_pause_count = _safe_record_int(record.get("host_pause_count", 0))
    # The source resets this legacy field on every recovery and compatibility
    # view; it never forces a retry to CPU.
    runtime.force_cpu_after_cuda_oom = False
    runtime.exit_code = record.get("exit_code")
    runtime.last_exit_code = record.get("last_exit_code", record.get("exit_code"))
    runtime.started_at = _safe_record_float(record.get("started_at", 0.0))
    raw_ended_at = record.get("ended_at")
    runtime.ended_at = None if raw_ended_at is None else _safe_record_float(raw_ended_at)
    runtime.last_error = record.get("last_error")
    saved_context = record.get("launch_context")
    runtime.launch_context = dict(saved_context) if isinstance(saved_context, dict) else {}
    runtime.recovered_from_status = record.get("recovered_from_status")
    runtime.recovered_at = record.get("recovered_at")
    saved_snapshot = record.get("resource_snapshot")
    runtime.resource_snapshot = dict(saved_snapshot) if isinstance(saved_snapshot, dict) else {}
    runtime_record_fields = {
        "job_id", "identity", "route", "task_ref", "command", "result_dir",
        "split_file", "attempt", "retry_count", "status", "pid", "device_mode",
        "pause_reason", "pause_count", "gpu_pause_count", "host_pause_count",
        "force_cpu_after_cuda_oom", "exit_code", "last_exit_code", "started_at",
        "ended_at", "last_progress_ts", "last_error", "launch_context",
        "resource_snapshot", "log_path", "updated_at", "state_path",
        "recovered_from_status", "recovered_at",
    }
    runtime.unknown_state_fields = {
        key: value for key, value in record.items() if key not in runtime_record_fields
    }
    # The source counts completed zero-exit records before placing reconciled
    # runtimes into active/suspended/queued collections.  A stale live PID on
    # such a record must not turn a completed route into an unknown-exit retry.
    if persisted_status == RunStatus.COMPLETED and _safe_record_int(record.get("last_exit_code", 0)) == 0:
        runtime.status = RunStatus.COMPLETED
        runtime.exit_code = 0
        return "completed"
    if (
        raw_status in {"failed", "fail"}
        and (record.get("last_error") or record.get("reason")) == "inexact_checkpoint_resume"
    ):
        runtime.status = RunStatus.FAILED
        return "terminal"
    # The source reconciliation deliberately clears a paused record's saved
    # PID, then performs a fresh exact-argv search in load_existing_runtime.
    # Do not trust a merely live PID here: it may have been reused by an
    # unrelated process after the paused child was terminated.
    if persisted_status == RunStatus.PAUSED:
        runtime.pid = None
        runtime.process = None
        runtime.recovered_process = False
        runtime.pause_reason = str(record.get("pause_reason") or "")
        runtime.exit_code = None
        runtime.last_exit_code = None
        runtime.ended_at = None
        runtime.status = RunStatus.PAUSED
        recovered_at = local_timestamp()
        runtime.recovered_from_status = RunStatus.PAUSED.value
        runtime.recovered_at = recovered_at
        if state_store is not None:
            # The source persists the paused/PID-cleared normalization before
            # its subsequent exact-command attachment attempt.
            _save_reconciled_state(runtime, state_store, recovered_at)
        matched = find_exact_argv(runtime.spec.command)
        if matched is not None:
            runtime.pid = int(matched.pid)
            runtime.process = RecoveredProcessHandle(runtime.pid)
            runtime.recovered_process = True
            runtime.status = RunStatus.RUNNING
            return "active"
        runtime.pid = None
        runtime.process = None
        runtime.status = RunStatus.PAUSED
        return "suspended"
    if runtime.pid is not None and process_is_live(runtime.pid):
        live_pid = runtime.pid
        runtime.process = RecoveredProcessHandle(live_pid)
        runtime.recovered_process = True
        runtime.status = RunStatus.RUNNING
        if persisted_status in {RunStatus.RUNNING, RunStatus.LAUNCHING, RunStatus.RETRYING}:
            same_running_pid = persisted_status == RunStatus.RUNNING and _safe_record_int(record.get("pid")) == live_pid
            if not same_running_pid:
                _persist_live_recovery(runtime, persisted_status, state_store)
        return "active"
    matched = find_exact_argv(runtime.spec.command)
    if matched is not None:
        runtime.pid = matched.pid
        runtime.process = RecoveredProcessHandle(matched.pid)
        runtime.recovered_process = True
        runtime.status = RunStatus.RUNNING
        if persisted_status in {RunStatus.RUNNING, RunStatus.LAUNCHING, RunStatus.RETRYING}:
            _persist_live_recovery(runtime, persisted_status, state_store)
        return "active"
    if persisted_status == RunStatus.COMPLETED and _safe_record_int(record.get("last_exit_code") or 0) == 0:
        runtime.status = RunStatus.COMPLETED
        runtime.exit_code = 0
        return "completed"
    runtime.pid = None
    runtime.process = None
    runtime.exit_code = None
    runtime.ended_at = None
    runtime.status = RunStatus.QUEUED
    return "queued"


def _safe_record_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _safe_record_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _persist_live_recovery(
    runtime: JobRuntime,
    previous_status: RunStatus,
    state_store: JobStateStore | None,
) -> None:
    """Normalize a recovered launch-stage record and persist its live PID."""
    recovered_at = local_timestamp()
    runtime.status = RunStatus.RUNNING
    runtime.pause_reason = ""
    runtime.exit_code = None
    runtime.ended_at = None
    runtime.recovered_from_status = previous_status.value
    runtime.recovered_at = recovered_at
    if state_store is not None:
        _save_reconciled_state(runtime, state_store, recovered_at)


def _save_reconciled_state(
    runtime: JobRuntime,
    state_store: JobStateStore,
    updated_at: str,
) -> None:
    """Avoid rebuilding compatibility views during source-style recovery."""
    save_state_only = getattr(state_store, "save_state_only", None)
    if callable(save_state_only):
        save_state_only(runtime, updated_at=updated_at)
    else:
        # Keep lightweight injected state stores usable in tests and plugins.
        state_store.save(runtime, updated_at=updated_at)
