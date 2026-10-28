"""Durable per-run state records and compatibility aggregation views."""

from __future__ import annotations

import csv
import json
import os
import shutil
from urllib.parse import quote
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .compat import ensure_dir, local_timestamp
from .plan import RunSpec


class RunStatus(StrEnum):
    QUEUED = "queued"
    LAUNCHING = "launching"
    RUNNING = "running"
    PAUSED = "paused"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


@dataclass(slots=True)
class JobRuntime:
    spec: RunSpec
    status: RunStatus = RunStatus.QUEUED
    attempt: int = 0
    retry_count: int = 0
    pid: int | None = None
    device_mode: str | None = None
    pause_reason: str | None = None
    pause_count: int = 0
    gpu_pause_count: int = 0
    host_pause_count: int = 0
    # Persisted for compatibility; scheduler retries do not use this as a
    # device override and recovery always clears it.
    force_cpu_after_cuda_oom: bool = False
    exit_code: int | None = None
    last_exit_code: int | None = None
    started_at: float = 0.0
    ended_at: float | None = None
    last_progress_ts: float = 0.0
    last_error: str | None = None
    launch_context: dict[str, Any] = field(default_factory=dict)
    recovered_process: bool = False
    recovered_from_status: str | None = None
    recovered_at: str | None = None
    log_path: str | None = None
    resource_snapshot: dict[str, Any] = field(default_factory=dict)
    unknown_state_fields: dict[str, Any] = field(default_factory=dict, repr=False)
    process: Any = field(default=None, repr=False, compare=False)
    log_handle: Any = field(default=None, repr=False, compare=False)

    @property
    def job_id(self) -> str:
        return self.spec.identity.run_id

    @property
    def is_gpu(self) -> bool:
        return self.device_mode == "gpu"

    @property
    def live(self) -> bool:
        if self.process is None:
            return False
        if self.recovered_process:
            return _recovered_pid_live(self.pid)
        try:
            return self.process.poll() is None
        except (AttributeError, OSError):
            return False

    @property
    def derived_status(self) -> RunStatus:
        # The reference runtime derives status from its retained last exit
        # code, not a transient/current-attempt exit-code field.
        if self.last_exit_code == 0 and self.ended_at:
            return RunStatus.COMPLETED
        if self.live:
            return RunStatus.RUNNING
        if self.status == RunStatus.PAUSED:
            return RunStatus.PAUSED
        if self.last_exit_code is None:
            return RunStatus.QUEUED
        # The final branch classifies any retained zero exit code as completed,
        # even when ended_at is empty.
        return RunStatus.FAILED if self.last_exit_code != 0 else RunStatus.COMPLETED

    def state_path(self) -> Path:
        return Path(self.spec.identity.run_id) / "job_state.json"

    def to_record(self) -> dict[str, Any]:
        record = {
            **self.unknown_state_fields,
            "job_id": self.job_id,
            "identity": self.spec.identity.as_dict(),
            "route": self.spec.route.as_dict(),
            "task_ref": asdict(self.spec.task_ref),
            "command": list(self.spec.command),
            "result_dir": self.spec.result_dir,
            "split_file": self.spec.split_file,
            "attempt": self.attempt,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "pid": self.pid,
            "device_mode": self.device_mode,
            "pause_reason": self.pause_reason,
            "pause_count": self.pause_count,
            "gpu_pause_count": self.gpu_pause_count,
            "host_pause_count": self.host_pause_count,
            "force_cpu_after_cuda_oom": self.force_cpu_after_cuda_oom,
            "exit_code": self.exit_code,
            "last_exit_code": self.last_exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "last_progress_ts": self.last_progress_ts,
            "last_error": self.last_error,
            "launch_context": self.launch_context,
            "resource_snapshot": self.resource_snapshot,
            "log_path": self.log_path,
            "updated_at": local_timestamp(),
        }
        if self.recovered_from_status is not None:
            record["recovered_from_status"] = self.recovered_from_status
        if self.recovered_at is not None:
            record["recovered_at"] = self.recovered_at
        return record


def _recovered_pid_live(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        process = __import__("psutil").Process(pid)
        return process.is_running() and process.status() != __import__("psutil").STATUS_ZOMBIE
    except Exception:
        return False


class JobStateStore:
    """Per-run atomic JSON state, lifecycle events, and completion progress."""

    EVENT_FIELDS = (
        "timestamp", "event", "job_id", "status", "attempt", "retry_count",
        "device_mode", "pid", "exit_code", "reason", "result_dir", "resource_snapshot",
    )
    PROGRESS_FIELDS = (
        "timestamp", "event", "job_id", "provider", "suite", "task_id",
        "split_repeat", "fold", "sample", "suite_repeat", "model_id",
        "generic_profile", "route_label", "seed", "depth", "width",
        "architecture", "attempt", "device_mode", "exit_code", "reason",
        "results_dir", "resource_snapshot",
    )

    def __init__(self, run_root: str | Path) -> None:
        self.run_root = Path(run_root)
        self.jobs_root = self.run_root / "jobs"
        self.events_path = self.run_root / "events.csv"
        self.progress_path = self.run_root / "run_progress.csv"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, runtime: JobRuntime) -> Path:
        return self.jobs_root / runtime.job_id

    def job_dir_from_id(self, run_id: str) -> Path:
        return self.jobs_root / run_id

    def state_path(self, runtime: JobRuntime) -> Path:
        return self.job_dir(runtime) / "job_state.json"

    def save(self, runtime: JobRuntime, *, updated_at: str | None = None) -> None:
        self.save_state_only(runtime, updated_at=updated_at)
        self._write_compatibility_views(runtime)

    def save_state_only(self, runtime: JobRuntime, *, updated_at: str | None = None) -> None:
        """Persist only the per-job record, as restart reconciliation does."""
        record = runtime.to_record()
        record["state_path"] = str(self.state_path(runtime))
        if updated_at is not None:
            record["updated_at"] = updated_at
        atomic_write_json(self.state_path(runtime), record)

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self.jobs_root / run_id / "job_state.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def append_event(self, runtime: JobRuntime, event: str, *, reason: str | None = None, snapshot: dict[str, Any] | None = None, exit_code: int | None = None) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.events_path.exists()
        with self.events_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.EVENT_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": local_timestamp(), "event": event, "job_id": runtime.job_id,
                "status": runtime.status.value, "attempt": runtime.attempt,
                "retry_count": runtime.retry_count, "device_mode": runtime.device_mode,
                "pid": runtime.pid, "exit_code": runtime.exit_code if exit_code is None else exit_code,
                "reason": reason or "", "result_dir": runtime.spec.result_dir,
                "resource_snapshot": json.dumps(snapshot or runtime.resource_snapshot, sort_keys=True),
            })
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except Exception:
                pass

    def append_progress(self, runtime: JobRuntime, event: str, *, reason: str | None = None, snapshot: dict[str, Any] | None = None) -> None:
        """Append one terminal attempt row, mirroring the ledger's run_progress.csv."""
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.progress_path.exists()
        identity = runtime.spec.identity
        route = runtime.spec.route
        fixed_grid = route.fixed_grid
        if fixed_grid:
            depth = len(fixed_grid[0])
            width = max(fixed_grid[0], default=0)
            architecture: object = [list(widths) for widths in fixed_grid]
        else:
            depth = route.init_depth or runtime.spec.worker_settings.get("init_depth")
            width = route.init_width or runtime.spec.worker_settings.get("init_width")
            architecture = {"depth": depth, "width": width}
        with self.progress_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.PROGRESS_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": local_timestamp(),
                "event": event,
                "job_id": runtime.job_id,
                "provider": identity.provider,
                "suite": identity.suite,
                "task_id": identity.task_id,
                "split_repeat": identity.split_repeat,
                "fold": identity.fold,
                "sample": identity.sample,
                "suite_repeat": identity.suite_repeat,
                "model_id": route.model_id,
                "generic_profile": route.generic_profile or "",
                "route_label": route.route_label,
                "seed": identity.seed,
                "depth": depth if depth is not None else "",
                "width": width if width is not None else "",
                "architecture": json.dumps(architecture, sort_keys=True),
                "attempt": runtime.attempt,
                "device_mode": runtime.device_mode or "",
                "exit_code": runtime.last_exit_code if runtime.last_exit_code is not None else "",
                "reason": reason or "",
                "results_dir": runtime.spec.result_dir,
                "resource_snapshot": json.dumps(snapshot or runtime.resource_snapshot, sort_keys=True),
            })
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except Exception:
                pass

    def _write_compatibility_views(self, runtime: JobRuntime) -> None:
        directory = self.job_dir(runtime)
        identity = runtime.spec.identity.as_dict()
        route = runtime.spec.route
        group = self._compatibility_group(identity, route)
        successful = runtime.status == RunStatus.COMPLETED and runtime.last_exit_code == 0
        candidate_state = {
            "schema": 1,
            "scope": "candidate",
            "job_id": runtime.job_id,
            "identity": identity,
            "route": route.as_dict(),
            "task_ref": asdict(runtime.spec.task_ref),
            "candidate_dir": str(directory),
            "results_dir": runtime.spec.result_dir,
            "state_path": str(self.state_path(runtime)),
            "architecture": {
                "depth": len(route.fixed_grid[0]) if route.fixed_grid else route.init_depth,
                "width": max(route.fixed_grid[0]) if route.fixed_grid else route.init_width,
            },
            "status": runtime.status.value,
            "completed": successful,
            "attempt": runtime.attempt,
            "retry_count": runtime.retry_count,
            "device_mode": runtime.device_mode,
            "pause_count": runtime.pause_count,
            "gpu_pause_count": runtime.gpu_pause_count,
            "host_pause_count": runtime.host_pause_count,
            "force_cpu_after_cuda_oom": False,
            "pause_reason": runtime.pause_reason,
            "command": list(runtime.spec.command),
            "exit_code": runtime.exit_code,
            "last_exit_code": runtime.last_exit_code,
            "started_at": runtime.started_at,
            "ended_at": runtime.ended_at,
            "last_error": runtime.last_error,
            "launch_context": runtime.launch_context,
            "resource_snapshot": runtime.resource_snapshot,
            "updated_at": local_timestamp(),
        }
        atomic_write_json(directory / "candidate_state.json", candidate_state)
        atomic_write_json(directory / "candidate_summary.json", {
            "schema": 1,
            "job_id": runtime.job_id,
            "completed": successful,
            "status": runtime.status.value,
            "exit_code": runtime.last_exit_code,
            "reason": runtime.last_error,
            "attempt": runtime.attempt,
            "device_mode": runtime.device_mode,
            "ended_at": runtime.ended_at,
            "results_dir": runtime.spec.result_dir,
            "state_path": str(self.state_path(runtime)),
            "updated_at": local_timestamp(),
        })
        method_state, task_state, replay_manifest = self._summarize_compatibility_groups(
            identity, route, group,
        )
        atomic_write_json(group["method_root"] / "method_state.json", method_state)
        atomic_write_json(group["task_root"] / "task_state.json", task_state)
        atomic_write_json(group["task_root"] / "replay_manifest.json", replay_manifest)

    @staticmethod
    def _slug(value: object) -> str:
        normalized = str(value or "unknown").strip()
        return quote(normalized, safe="") or "unknown"

    def _compatibility_group(self, identity: dict[str, Any], route: Any) -> dict[str, Path | str | int | None]:
        provider = self._slug(identity.get("provider"))
        suite = self._slug(identity.get("suite"))
        task_id = self._slug(identity.get("task_id"))
        repeat = int(identity.get("suite_repeat", 1))
        task_root = self.run_root / "tasks" / provider / suite / f"task_{task_id}" / f"repeat_{repeat:02d}"
        profile = route.generic_profile or "default"
        method_key = f"{self._slug(route.model_id)}@{self._slug(profile)}"
        method_root = task_root / "methods" / method_key
        return {
            "task_root": task_root,
            "method_root": method_root,
            "provider": provider,
            "suite": suite,
            "task_id": task_id,
            "repeat": repeat,
            "method_key": method_key,
            "model_id": route.model_id,
            "generic_profile": route.generic_profile,
        }

    @staticmethod
    def _safe_exit_code(state: dict[str, Any]) -> int:
        try:
            # Compatibility summaries mirror the source record, where
            # last_exit_code is authoritative (a missing/unparseable value is
            # a failure, even if an unrelated current-attempt field says zero).
            return int(state.get("last_exit_code"))
        except Exception:
            return 1

    def _summarize_compatibility_groups(
        self,
        identity: dict[str, Any],
        route: Any,
        group: dict[str, Path | str | int | None],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        states: list[dict[str, Any]] = []
        state_paths = sorted(self.jobs_root.rglob("job_state.json")) if self.jobs_root.exists() else ()
        for state_path in state_paths:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(state, dict):
                state["state_path"] = str(state_path)
                states.append(state)

        def mapping_field(state: dict[str, Any], name: str) -> dict[str, Any]:
            value = state.get(name)
            return value if isinstance(value, dict) else {}

        def same_task(state: dict[str, Any]) -> bool:
            other = state.get("identity")
            if not isinstance(other, dict):
                return False
            return all(other.get(key) == identity.get(key) for key in ("provider", "suite", "task_id", "suite_repeat"))

        def same_method(state: dict[str, Any]) -> bool:
            if not same_task(state):
                return False
            other_route = state.get("route")
            if not isinstance(other_route, dict):
                return False
            return (
                other_route.get("model_id") == route.model_id
                and (other_route.get("generic_profile") or "default") == (route.generic_profile or "default")
            )

        method_states = [state for state in states if same_method(state)]
        task_states = [state for state in states if same_task(state)]

        def counts(values: list[dict[str, Any]]) -> dict[str, int]:
            result = {key: 0 for key in ("completed", "running", "paused", "failed", "queued")}
            for value in values:
                status = str(value.get("status") or "queued")
                if status == "launching":
                    result["running"] += 1
                elif status in result:
                    result[status] += 1
                else:
                    result["queued"] += 1
            return result

        def latest_completed(values: list[dict[str, Any]]) -> dict[str, Any] | None:
            completed = [
                value for value in values
                if value.get("status") == "completed" and self._safe_exit_code(value) == 0
            ]
            if not completed:
                return None

            def ended_key(value: dict[str, Any]) -> tuple[int, str, str]:
                ended = str(value.get("ended_at") or "")
                # Preserve the copied helper's permissive predicate exactly:
                # remove only the first decimal point, then use str.isdigit().
                is_numeric = ended.replace(".", "", 1).isdigit()
                return int(is_numeric), ended, str(value.get("job_id") or "")

            return sorted(completed, key=ended_key)[-1]

        success = lambda value: value.get("status") == "completed" and self._safe_exit_code(value) == 0

        def family_key(state: dict[str, Any]) -> tuple[str, str] | None:
            route_record = mapping_field(state, "route")
            model_id = route_record.get("model_id")
            profile = route_record.get("generic_profile")
            if not isinstance(model_id, str) or not model_id:
                return None
            if profile is not None and not isinstance(profile, str):
                return None
            return model_id, profile or "default"

        method_root = group["method_root"]
        task_root = group["task_root"]
        method_state = {
            "schema": 1,
            "scope": "method",
            "provider": group["provider"],
            "suite": group["suite"],
            "task_id": group["task_id"],
            "repeat": group["repeat"],
            "model_id": route.model_id,
            "generic_profile": route.generic_profile,
            "candidate_count": len(method_states),
            "counts": counts(method_states),
            "latest_completed": latest_completed(method_states),
            "candidate_root": str(method_root),
            "candidate_states": [state["state_path"] for state in method_states if state.get("state_path")],
            "best_candidate_dir": next((state.get("result_dir") for state in method_states if success(state)), None),
            "updated_at": local_timestamp(),
        }
        method_roots = sorted({
            str(self._compatibility_group(
                mapping_field(state, "identity"), _RouteRecord(mapping_field(state, "route")),
            )["method_root"])
            for state in task_states
            if (
                isinstance(state.get("identity"), dict)
                and isinstance(state.get("route"), dict)
                and family_key(state) is not None
            )
        })
        task_state = {
            "schema": 1,
            "scope": "task",
            "provider": group["provider"],
            "suite": group["suite"],
            "task_id": group["task_id"],
            "repeat": group["repeat"],
            "family_count": len({
                key for state in task_states if (key := family_key(state)) is not None
            }),
            "candidate_count": len(task_states),
            "counts": counts(task_states),
            "latest_completed": latest_completed(task_states),
            "task_root": str(task_root),
            "method_roots": method_roots,
            "best_candidate_dir": next((state.get("result_dir") for state in task_states if success(state)), None),
            "updated_at": local_timestamp(),
        }
        replay_manifest = {
            "schema": 1,
            "provider": group["provider"],
            "suite": group["suite"],
            "task_id": group["task_id"],
            "repeat": group["repeat"],
            "seed": identity.get("seed"),
            "task_root": str(task_root),
            "method_root": str(method_root),
            "candidate_dir": str(self.job_dir_from_id(str(identity.get("run_id", "")))),
            "candidates": [
                {
                    "job_id": state.get("job_id"),
                    "model_id": mapping_field(state, "route").get("model_id"),
                    "generic_profile": mapping_field(state, "route").get("generic_profile"),
                    "route_label": mapping_field(state, "route").get("route_label"),
                    "seed": mapping_field(state, "identity").get("seed"),
                    "candidate_dir": state.get("result_dir"),
                    "state_path": state.get("state_path"),
                    "status": state.get("status"),
                    "attempt": state.get("attempt"),
                    "device_mode": state.get("device_mode"),
                    "exit_code": state.get("last_exit_code"),
                }
                for state in task_states
            ],
            "updated_at": local_timestamp(),
        }
        return method_state, task_state, replay_manifest


class _RouteRecord:
    """Small route view used only while deriving grouped method paths."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.model_id = str(payload.get("model_id") or "unknown")
        self.generic_profile = payload.get("generic_profile")


def seed_run_root_from_source(source: str | Path | None, destination: str | Path) -> dict[str, int]:
    """Copy source run artifacts without overwriting; no checksum or merge."""
    if not source:
        return {"copied": 0, "skipped": 0}
    source_root, destination_root = Path(source), Path(destination)
    if not source_root.exists():
        return {"copied": 0, "skipped": 0}
    named = (
        "manifest.json", "job_manifest.json", "run_metadata.json", "batch_backoff_state.json",
        "run_progress.csv", "task_summary.csv", "final_report.json", "final_report.md",
        "suite_summary.json", "suite_summary.md",
    )
    copied = skipped = 0

    def copy_one(src: Path, dst: Path) -> None:
        nonlocal copied, skipped
        if dst.exists():
            skipped += 1
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    for filename in named:
        src = source_root / filename
        if not src.exists():
            continue
        copy_one(src, destination_root / filename)
    for src in source_root.rglob("*"):
        if not src.is_file():
            continue
        relative_path = src.relative_to(source_root)
        if relative_path.parts and relative_path.parts[0] in named:
            continue
        copy_one(src, destination_root / relative_path)
    return {"copied": copied, "skipped": skipped}
