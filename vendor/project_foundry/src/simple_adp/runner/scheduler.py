"""Central pressure-aware suite controller.

The controller is a fresh generic implementation.  It consumes the public
provider/model adapters and never imports code from the copied reference tree.
Its ordering and intentionally source-compatible edge behavior are documented
in ``docs/COORDINATED_PRESSURE_AWARE_RUNNER_LEDGER.md``.
"""

from __future__ import annotations

import json
import csv
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .admission import (
    AdmissionState,
    LaneCounts,
    choose_device,
    hard_pressure,
    launch_budget as scaled_launch_budget,
    pressure_pause_reopened,
    reopen_external_gates,
    resolve_launch_capacity,
    update_soft_gates,
)
from .config import RunnerSettings, UNBOUNDED_CONCURRENCY
from .artifacts import estimator_checkpoint_path
from .logging_context import install_resource_record_factory
from .processes import (
    RecoveredProcessHandle,
    close_log_handle,
    launch_job,
    reconcile_runtime,
    terminate_process_tree,
)
from .resources import (
    CallbackResourceProvider,
    ProcessMemorySample,
    ResourceSnapshot,
    ResourceProvider,
    external_drop_reopened,
    process_tree_pids,
    sample_child_memory,
    sample_active_rss_mib,
    sample_external_gpu_memory,
    sample_external_host_memory,
    sample_gpu_memory,
    sample_host_memory,
)
from .state import (
    JobRuntime,
    JobStateStore,
    RunStatus,
    atomic_write_json,
    atomic_write_text,
    seed_run_root_from_source,
    utc_now,
)
from .telemetry import summarize_live
from .screening import build_screening_reports

LOG = logging.getLogger("simple_adp.runner")
StateSampler = Callable[[int], Any]


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _summary_markdown(summary: dict[str, Any], *, persisted_state: bool, include_timestamp: bool) -> str:
    retry_label = "Retries recorded in job state" if persisted_state else "Retries"
    lines = [
        "# Suite report",
        "",
        f"- Run root: `{summary['run_root']}`",
        f"- Results directory: `{summary['results_dir']}`",
        f"- Total jobs: {summary['manifest_jobs']}",
        f"- Completed: {summary['completed']}",
        f"- Terminal failed: {summary['terminal_failures']}",
        f"- Failed attempts: {summary.get('failed_attempts', 'unknown')}",
        f"- {retry_label}: {summary['retries']}",
        f"- Batches done: {summary['batches_done']}",
        f"- Epochs done: {summary['epochs_done']}",
        f"- Live-job batches done: {summary.get('live_batches_done', 0)}",
        f"- Live-job epochs done: {summary.get('live_epochs_done', 0)}",
    ]
    if persisted_state:
        lines.append("- Report mode: persisted state snapshot")
    if include_timestamp:
        lines.append(f"- Timestamp: {summary['timestamp']}")
    return "\n".join(lines) + "\n"


@dataclass(slots=True)
class InvocationCounters:
    completed: int = 0
    failed_attempts: int = 0
    terminal_failures: int = 0
    retries: int = 0
    launched: int = 0
    recovered_unknown_exit: int = 0


@dataclass(slots=True)
class ControllerMemory:
    """Persisted controller record; transient fields reset at each invocation."""

    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def defaults(cls, settings: RunnerSettings) -> "ControllerMemory":
        return cls({
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
            "host_gate_buffer_mib": settings.child_host_gate_buffer_mib,
            "gpu_gate_buffer_mib": settings.child_gpu_gate_buffer_mib,
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
        })

    def reset_for_invocation(self, settings: RunnerSettings) -> None:
        self.values.update(self.defaults(settings).values)

    def admission_state(self) -> AdmissionState:
        return AdmissionState(
            gpu_gate_closed=bool(self.values.get("gpu_gate_closed", False)),
            host_gate_closed=bool(self.values.get("host_gate_closed", False)),
            gpu_gate_lock_job_id=self.values.get("gpu_gate_lock_job_id"),
            host_gate_lock_job_id=self.values.get("host_gate_lock_job_id"),
            gpu_launches_enabled=bool(self.values.get("gpu_launches_enabled", True)),
            cpu_launches_enabled=bool(self.values.get("cpu_launches_enabled", True)),
            gpu_hold_until=float(self.values.get("gpu_hold_until", 0.0)),
            cpu_hold_until=float(self.values.get("host_hold_until", 0.0)),
            batch_scale=float(self.values.get("batch_scale", 1.0)),
            pressure_backoff_pending=bool(self.values.get("pressure_backoff_pending", False)),
            last_backoff_reason=self.values.get("last_backoff_reason"),
        )

    def update_admission(self, state: AdmissionState) -> None:
        self.values.update({
            "gpu_gate_closed": state.gpu_gate_closed,
            "host_gate_closed": state.host_gate_closed,
            "gpu_gate_lock_job_id": state.gpu_gate_lock_job_id,
            "host_gate_lock_job_id": state.host_gate_lock_job_id,
            "gpu_launches_enabled": state.gpu_launches_enabled,
            "cpu_launches_enabled": state.cpu_launches_enabled,
            "gpu_hold_until": state.gpu_hold_until,
            "host_hold_until": state.cpu_hold_until,
            "batch_scale": state.batch_scale,
            "pressure_backoff_pending": state.pressure_backoff_pending,
            "last_backoff_reason": state.last_backoff_reason,
        })


class SuiteCoordinator:
    """Run ordered one-child-per-route jobs under live resource pressure."""

    def __init__(
        self,
        plan: Any,
        settings: RunnerSettings,
        *,
        run_root: str | Path,
        results_dir: str | Path,
        repository_root: str | Path,
        source_run_root: str | Path | None = None,
        host_sampler: Callable[[], Any] = sample_host_memory,
        gpu_sampler: Callable[[int], Any] = sample_gpu_memory,
        resource_provider: ResourceProvider | None = None,
        poll_sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.plan = plan
        self.settings = settings
        self.run_root = Path(run_root).resolve()
        self.results_dir = Path(results_dir).resolve()
        self.repository_root = Path(repository_root).resolve()
        self.source_run_root = Path(source_run_root) if source_run_root else None
        self.resource_provider = resource_provider or CallbackResourceProvider(
            host_sampler=host_sampler,
            gpu_sampler=gpu_sampler,
            tree_sampler=process_tree_pids,
            child_sampler=sample_child_memory,
            external_host_sampler=sample_external_host_memory,
            external_gpu_sampler=sample_external_gpu_memory,
        )
        self.sleep = poll_sleep
        self.clock = wall_clock
        self.state_store = JobStateStore(self.run_root)
        self.memory = ControllerMemory.defaults(settings)
        self.admission = AdmissionState()
        self.counters = InvocationCounters()
        self.last_resource_snapshot: dict[str, Any] = {}
        self.logging_counts = LaneCounts(total=0, gpu=0, cpu=0)
        install_resource_record_factory(self._logging_resource_context)
        self.last_status_log_ts = 0.0
        self.shutdown_signal: int | None = None
        self._previous_handlers: dict[int, Any] = {}

    def _logging_resource_context(self) -> dict[str, Any]:
        """Expose the last completed resource sample to subsequent log records."""
        snapshot = self.last_resource_snapshot
        host = snapshot.get("host") or {}
        gpu = snapshot.get("gpu") or {}
        return {
            "host_total_mib": float(host.get("total_mib", 0.0)),
            "host_used_mib": float(host.get("total_mib", 0.0) - host.get("available_mib", 0.0)),
            "host_available_mib": float(host.get("available_mib", 0.0)),
            "gpu_used_mib": float(gpu.get("used_mib", 0.0)),
            "gpu_total_mib": float(gpu.get("total_mib", 0.0)),
            "gpu_processes": self.logging_counts.gpu,
            "cpu_processes": self.logging_counts.cpu,
        }

    @property
    def batch_state_path(self) -> Path:
        return self.run_root / "batch_backoff_state.json"

    def _load_controller_state(self) -> None:
        try:
            value = json.loads(self.batch_state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                defaults = ControllerMemory.defaults(self.settings).values
                defaults.update(value)
                self.memory = ControllerMemory(defaults)
        except Exception:
            self.memory = ControllerMemory.defaults(self.settings)
        self.memory.reset_for_invocation(self.settings)
        self.admission = self.memory.admission_state()
        self._save_controller_state()

    def _save_controller_state(self) -> None:
        self.memory.update_admission(self.admission)
        atomic_write_json(self.batch_state_path, self.memory.values)

    def _build_runtimes(self) -> tuple[list[JobRuntime], list[JobRuntime], deque[JobRuntime]]:
        active: list[JobRuntime] = []
        suspended: list[JobRuntime] = []
        queue: deque[JobRuntime] = deque()
        for spec in self.plan.runs:
            runtime = JobRuntime(spec)
            location = reconcile_runtime(
                runtime,
                self.state_store.load(runtime.job_id),
                state_store=self.state_store,
            )
            if location == "completed":
                self.counters.completed += 1
            elif location == "terminal":
                self.counters.terminal_failures += 1
            elif location == "active":
                active.append(runtime)
            elif location == "suspended":
                suspended.append(runtime)
            else:
                queue.append(runtime)
        return active, suspended, queue

    @staticmethod
    def _counts(active: list[JobRuntime]) -> LaneCounts:
        live = [runtime for runtime in active if runtime.live]
        return LaneCounts(len(live), sum(runtime.is_gpu for runtime in live), sum(not runtime.is_gpu for runtime in live))

    def _record_event(
        self,
        runtime: JobRuntime,
        event: str,
        *,
        reason: str | None = None,
        snapshot: dict[str, Any] | None = None,
        exit_code: int = 0,
        log_event: bool = True,
    ) -> None:
        """Persist an event and emit its source-shaped operational log row."""
        if log_event:
            route = runtime.spec.route
            if route.fixed_grid is not None:
                depth = len(route.fixed_grid)
                width = max((max(layer, default=0) for layer in route.fixed_grid), default=0)
            else:
                depth = int(route.init_depth or runtime.spec.worker_settings.get("init_depth", 1))
                width = int(route.init_width or runtime.spec.worker_settings.get("init_width", 1))
            LOG.info(
                "[%s] %s case=%s repeat=%s phase=%s family=%s depth=%s width=%s attempt=%s device=%s reason=%s exit=%s",
                event.upper(),
                runtime.job_id,
                runtime.spec.task_ref.task_id,
                runtime.spec.identity.suite_repeat,
                route.suite_group or "model",
                route.model_id,
                depth,
                width,
                runtime.attempt,
                runtime.device_mode,
                reason or "-",
                exit_code,
            )
        self.state_store.append_event(
            runtime,
            event,
            reason=reason,
            snapshot=snapshot,
            exit_code=exit_code,
        )

    @staticmethod
    def _latest(active: list[JobRuntime], *, gpu_only: bool = False) -> JobRuntime | None:
        values = [runtime for runtime in active if runtime.live and (not gpu_only or runtime.is_gpu)]
        return sorted(values, key=lambda runtime: (runtime.started_at, runtime.attempt, runtime.job_id))[-1] if values else None

    def _sample_runtime_child(self, runtime: JobRuntime) -> ProcessMemorySample:
        """Recheck process liveness at the child-memory sampling boundary."""
        if runtime.process is None or not runtime.live:
            return ProcessMemorySample()
        return self.resource_provider.sample_child(
            runtime.pid,
            gpu_device_index=self.settings.gpu_device_index,
            gpu_mode=runtime.is_gpu,
        )

    def _status(self, active: list[JobRuntime], suspended: list[JobRuntime], queue: deque[JobRuntime], *, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self.last_status_log_ts < 30.0:
            return
        self.last_status_log_ts = now
        payload = {
            "timestamp": utc_now(),
            "run_signature": self.plan.run_signature,
            "queue": len(queue),
            "active": len([runtime for runtime in active if runtime.live]),
            "suspended": len(suspended),
            "completed": self.counters.completed,
            "terminal_failures": self.counters.terminal_failures,
            "retries": self.counters.retries,
            "resource_snapshot": self.last_resource_snapshot,
            **summarize_live(self.run_root, self.plan.runs, self.state_store),
        }
        atomic_write_json(self.run_root / "status.json", payload)
        resource = self.last_resource_snapshot
        host_sample = resource.get("host")
        gpu_sample = resource.get("gpu")
        host_sample = host_sample if isinstance(host_sample, dict) else {}
        gpu_sample = gpu_sample if isinstance(gpu_sample, dict) else {}
        LOG.info(
            "[STATUS] queued=%d active=%d suspended=%d completed=%d failed_attempts=%d terminal_failed=%d retried=%d batches_done=%d epochs_done=%d live_batches_done=%d live_epochs_done=%d telemetry_jobs=%d telemetry_dirs=%d batch_scale=%.2f gpu_launches=%s cpu_launches=%s gpu_gate_closed=%s host_gate_closed=%s gpu_hold=%.1fs cpu_hold=%.1fs host_avail=%s host_rss=%s gpu_free=%s gpu_used=%s last_child_rss=%s last_child_vram=%s launch_cap=%s",
            len(queue),
            len([runtime for runtime in active if runtime.live]),
            len(suspended),
            self.counters.completed,
            self.counters.failed_attempts,
            self.counters.terminal_failures,
            self.counters.retries,
            payload["batches_done"],
            payload["epochs_done"],
            payload["live_batches_done"],
            payload["live_epochs_done"],
            payload["telemetry_jobs"],
            payload["telemetry_dirs"],
            float(self.admission.batch_scale),
            bool(self.admission.gpu_launches_enabled),
            bool(self.admission.cpu_launches_enabled),
            bool(self.admission.gpu_gate_closed),
            bool(self.admission.host_gate_closed),
            max(0.0, float(self.admission.gpu_hold_until - now)),
            max(0.0, float(self.admission.cpu_hold_until - now)),
            host_sample.get("available_mib", "-"),
            resource.get("active_rss_mib", "-"),
            gpu_sample.get("free_mib", "-"),
            gpu_sample.get("used_mib", "-"),
            self.memory.values.get("last_child_host_mib", "-"),
            self.memory.values.get("last_child_gpu_mib", "-"),
            resource.get("launch_capacity", "-"),
        )

    def _sample_snapshot(self, active: list[JobRuntime], suspended: list[JobRuntime], concurrency: int) -> tuple[ResourceSnapshot, JobRuntime | None, JobRuntime | None]:
        host = self.resource_provider.sample_host()
        gpu = self.resource_provider.sample_gpu(self.settings.gpu_device_index)
        counts = self._counts(active)
        self.logging_counts = counts
        active_live = [runtime for runtime in active if runtime.live]
        active_pids: set[int] = set()
        suspended_pids: set[int] = set()
        for runtimes, destination in ((active, active_pids), (suspended, suspended_pids)):
            for runtime in runtimes:
                # The source PID helper includes only process handles that
                # still poll as live. Paused/stale runtimes commonly have no PID.
                if runtime.process is None or runtime.pid is None or not runtime.live:
                    continue
                try:
                    destination.update(self.resource_provider.process_tree(runtime.pid))
                except Exception:
                    # current_pids() swallows a process-tree inspection error
                    # and continues with the remaining owned runtimes.
                    continue
        sample_rss = getattr(self.resource_provider, "sample_active_rss", sample_active_rss_mib)
        try:
            # The reference performs one sum over active PIDs; any lookup
            # error makes that diagnostic total zero rather than partial.
            active_rss = float(sample_rss(active_pids))
        except Exception:
            active_rss = 0.0
        # The source's capacity helper rereads the live concurrency file here;
        # the loop-level value remains the launch-budget input from earlier in
        # this pass.
        capacity_concurrency = self.settings.resolve_concurrency()
        capacity = resolve_launch_capacity(
            self.settings,
            concurrency=capacity_concurrency,
            host_available_mib=host.available_mib,
            gpu=gpu,
            # The reference applies this ceiling in a separate conditional
            # after resolve_launch_capacity, with another concurrency read.
            active_count=0,
        )
        if counts.total > 0 and self.settings.resolve_concurrency() < UNBOUNDED_CONCURRENCY:
            capacity = max(1, min(capacity, counts.total + 1))
        # Strict pressure flags follow capacity calculation in the source.
        host_pressure, gpu_pressure = hard_pressure(self.settings, host, gpu)
        latest_all = self._latest(active)
        latest_gpu = self._latest(active, gpu_only=True)
        # Source-compatible sampling order: a live general latest runtime is
        # sampled first, so the separate latest-GPU fallback is unreachable.
        if latest_all is not None:
            latest_sample = self._sample_runtime_child(latest_all)
        elif latest_gpu is not None:
            latest_sample = self._sample_runtime_child(latest_gpu)
        else:
            latest_sample = None
        latest_host = latest_sample.host_rss_mib if latest_sample is not None else 0.0
        latest_gpu_mib = latest_sample.gpu_vram_mib if latest_sample is not None else 0.0

        previous_host = float(self.memory.values.get("external_host_mib", 0.0))
        previous_gpu = float(self.memory.values.get("external_gpu_mib", 0.0))
        previous_host_pids = self.memory.values.get("external_host_pids", [])
        previous_gpu_pids = self.memory.values.get("external_gpu_pids", [])
        excluded_pids = active_pids | suspended_pids
        host_external = self.resource_provider.sample_external_host(excluded_pids)
        gpu_external = self.resource_provider.sample_external_gpu(self.settings.gpu_device_index, excluded_pids)
        host_drop = previous_host - host_external.host_rss_mib
        gpu_drop = previous_gpu - gpu_external.gpu_vram_mib
        host_external_reopened = external_drop_reopened(
            previous_mib=previous_host,
            current_mib=host_external.host_rss_mib,
            previous_pids=previous_host_pids,
            current_pids=host_external.pid_set,
            threshold_mib=self.settings.external_memory_drop_mib,
        )
        gpu_external_reopened = external_drop_reopened(
            previous_mib=previous_gpu,
            current_mib=gpu_external.gpu_vram_mib,
            previous_pids=previous_gpu_pids,
            current_pids=gpu_external.pid_set,
            threshold_mib=self.settings.external_memory_drop_mib,
        )
        self.memory.values.update({
            "external_host_mib": host_external.host_rss_mib,
            "external_gpu_mib": gpu_external.gpu_vram_mib,
            "external_host_pids": sorted(host_external.pid_set),
            "external_gpu_pids": sorted(gpu_external.pid_set),
            "external_host_peak_mib": max(float(self.memory.values.get("external_host_peak_mib", 0.0)), host_external.host_rss_mib),
            "external_gpu_peak_mib": max(float(self.memory.values.get("external_gpu_peak_mib", 0.0)), gpu_external.gpu_vram_mib),
        })
        snapshot = ResourceSnapshot(
            host=host,
            gpu=gpu,
            active_count=counts.total,
            active_gpu_count=counts.gpu,
            active_cpu_count=counts.cpu,
            active_rss_mib=active_rss,
            latest_child_rss_mib=latest_host,
            latest_child_vram_mib=latest_gpu_mib,
            external_host_mib=host_external.host_rss_mib,
            external_gpu_mib=gpu_external.gpu_vram_mib,
            external_host_pids=host_external.pid_set,
            external_gpu_pids=gpu_external.pid_set,
            host_pressure=host_pressure,
            gpu_pressure=gpu_pressure,
            launch_capacity=capacity,
        )
        self.memory.values.update({
            "last_child_host_mib": latest_host,
            "last_child_gpu_mib": latest_gpu_mib,
            "last_host_remaining_mib": max(0.0, host.available_mib - latest_host),
            "last_gpu_remaining_mib": max(0.0, gpu.free_mib - latest_gpu_mib) if gpu is not None else 0.0,
            "last_child_job_id": latest_all.job_id if latest_all is not None else None,
        })
        # The report and logging context describe the resource sample before
        # this pass mutates gate state, matching the ledger's source ordering.
        self.last_resource_snapshot = {
            **snapshot.as_dict(),
            "concurrency": concurrency,
            "batch_scale": self.admission.batch_scale,
            "batch_launch_budget": scaled_launch_budget(self.settings, concurrency, self.admission.batch_scale),
            "host_gate_closed": self.admission.host_gate_closed,
            "gpu_gate_closed": self.admission.gpu_gate_closed,
            "host_gate_lock_job_id": self.admission.host_gate_lock_job_id,
            "gpu_gate_lock_job_id": self.admission.gpu_gate_lock_job_id,
            "host_launches_enabled": self.admission.cpu_launches_enabled,
            "gpu_launches_enabled": self.admission.gpu_launches_enabled,
            "host_hold_until": self.admission.cpu_hold_until,
            "gpu_hold_until": self.admission.gpu_hold_until,
            "external_host_drop_mib": host_drop,
            "external_gpu_drop_mib": gpu_drop,
            "external_host_reopened": host_external_reopened,
            "external_gpu_reopened": gpu_external_reopened,
            "latest_child_host_remaining_mib": max(0.0, host.available_mib - latest_host),
            "latest_child_gpu_remaining_mib": max(0.0, gpu.free_mib - latest_gpu_mib) if gpu is not None else None,
            "latest_child_job_id": latest_all.job_id if latest_all is not None else None,
        }
        gpu_gate_was_closed = self.admission.gpu_gate_closed
        host_gate_was_closed = self.admission.host_gate_closed
        update_soft_gates(
            self.admission,
            settings=self.settings,
            host_pressure=host_pressure,
            gpu_pressure=gpu_pressure,
            gpu_sample_present=gpu is not None,
            latest_gpu_job=latest_gpu.job_id if latest_gpu else None,
            latest_host_job=latest_all.job_id if latest_all else None,
            latest_child_gpu_mib=latest_gpu_mib,
            latest_child_host_mib=latest_host,
            host_available_mib=host.available_mib,
            gpu_free_mib=gpu.free_mib if gpu else None,
        )
        # Match the source controller's transition logs and ordering: report
        # newly closed gates first, then report any external-drop reopen from
        # this same sample.
        if not gpu_gate_was_closed and self.admission.gpu_gate_closed:
            gpu_buffer = float(self.settings.child_gpu_gate_buffer_mib)
            gpu_child_pressure = (
                latest_gpu_mib + gpu_buffer > float(gpu.free_mib if gpu else 0.0)
            )
            LOG.info(
                "[GATE] gpu admission closed job=%s child_vram=%.1f remaining_vram=%.1f buffer=%.1f gpu_pressure=%s child_pressure=%s",
                latest_gpu.job_id if latest_gpu else None,
                latest_gpu_mib,
                float(gpu.free_mib if gpu else 0.0),
                gpu_buffer,
                bool(gpu_pressure),
                bool(gpu_child_pressure),
            )
        if not host_gate_was_closed and self.admission.host_gate_closed:
            host_buffer = float(self.settings.child_host_gate_buffer_mib)
            host_child_pressure = latest_host + host_buffer > float(host.available_mib)
            LOG.info(
                "[GATE] host admission closed job=%s child_rss=%.1f remaining_host=%.1f buffer=%.1f host_pressure=%s child_pressure=%s",
                latest_all.job_id if latest_all else None,
                latest_host,
                float(host.available_mib),
                host_buffer,
                bool(host_pressure),
                bool(host_child_pressure),
            )
        gpu_reopened = gpu_external_reopened and self.admission.gpu_gate_closed
        host_reopened = host_external_reopened and self.admission.host_gate_closed
        reopen_external_gates(self.admission, host_drop=host_external_reopened, gpu_drop=gpu_external_reopened)
        if gpu_reopened and not self.admission.gpu_gate_closed:
            LOG.info(
                "[GATE] gpu admission reopened by external non-python memory drop host_free=%s gpu_free=%s",
                int(host.available_mib),
                int(gpu.free_mib) if gpu is not None else -1,
            )
        if host_reopened and not self.admission.host_gate_closed:
            LOG.info(
                "[GATE] host admission reopened by external non-python memory drop host_free=%d",
                int(host.available_mib),
            )
        self._save_controller_state()
        return snapshot, latest_all, latest_gpu

    def _resume_ready(self, runtime: JobRuntime, snapshot: ResourceSnapshot, host_drop: float, gpu_drop: float) -> bool:
        reason = runtime.pause_reason or "pressure"
        if not pressure_pause_reopened(
            pause_reason=reason,
            device=runtime.device_mode or "gpu",
            host_available_mib=snapshot.host.available_mib,
            gpu_free_mib=snapshot.gpu.free_mib if snapshot.gpu else None,
            host_used_mib=snapshot.host.total_mib - snapshot.host.available_mib,
            gpu_used_mib=snapshot.gpu.used_mib if snapshot.gpu else None,
            settings=self.settings,
            host_peak_used_mib=float(self.memory.values.get("cpu_peak_host_mib", 0.0)),
            gpu_peak_used_mib=float(self.memory.values.get("gpu_peak_vram_mib", 0.0)),
            external_host_drop_mib=host_drop,
            external_gpu_drop_mib=gpu_drop,
        ):
            return False
        # The source has this additional unconditional matching-lane resume gate.
        if runtime.device_mode == "gpu":
            return snapshot.gpu is not None and snapshot.gpu.free_mib >= self.settings.gpu_resume_avail_mib
        return snapshot.host.available_mib >= self.settings.host_ram_resume_avail_mib

    def _pause(self, runtime: JobRuntime, active: list[JobRuntime], suspended: list[JobRuntime], reason: str, snapshot: ResourceSnapshot) -> bool:
        if runtime.process is None or not runtime.live or runtime.status == RunStatus.PAUSED:
            return False
        try:
            terminate_process_tree(runtime.process)
        except Exception:
            return False
        runtime.status = RunStatus.PAUSED
        runtime.pause_reason = reason
        runtime.pause_count += 1
        lowered_reason = reason.lower()
        if "gpu" in lowered_reason or "cuda" in lowered_reason:
            runtime.gpu_pause_count += 1
        if "host" in lowered_reason or "swap" in lowered_reason:
            runtime.host_pause_count += 1
        runtime.last_progress_ts = self.clock()
        runtime.exit_code = None
        runtime.last_exit_code = None
        runtime.ended_at = None
        runtime.resource_snapshot = snapshot.as_dict()
        if runtime.log_handle is not None:
            close_log_handle(runtime.log_handle)
            runtime.log_handle = None
        runtime.process = None
        runtime.pid = None
        runtime.recovered_process = False
        if runtime in active:
            active.remove(runtime)
        if runtime not in suspended:
            suspended.append(runtime)
        self.state_store.save(runtime)
        self._record_event(runtime, "pause", reason=reason, snapshot=snapshot.as_dict())
        return True

    def _resume(self, runtime: JobRuntime, active: list[JobRuntime], snapshot: ResourceSnapshot, concurrency: int) -> bool:
        if runtime.process is not None and runtime.live:
            try:
                terminate_process_tree(runtime.process)
            except Exception:
                pass
        previous_reason = runtime.pause_reason or "pressure"
        runtime.status = RunStatus.QUEUED
        runtime.pause_reason = ""
        runtime.exit_code = None
        runtime.last_exit_code = None
        runtime.last_progress_ts = self.clock()
        try:
            # Resumes deliberately bypass capacity, gates, holds, and active caps.
            job_dir = self.state_store.job_dir(runtime)
            launch_job(
                runtime,
                settings=self.settings,
                state_store=self.state_store,
                repository_root=self.repository_root,
                launch_budget=concurrency,
                device=runtime.device_mode if runtime.device_mode in {"gpu", "cpu"} else "gpu",
                launch_context={
                    "resumed_from_pause": True,
                    "pause_reason": previous_reason,
                    "resume_device": runtime.device_mode if runtime.device_mode in {"gpu", "cpu"} else "gpu",
                    "checkpoint_last": str(job_dir / "checkpoint_last.pt"),
                    "checkpoint_best": str(job_dir / "checkpoint_best.pt"),
                    "checkpoint_best_batch": str(job_dir / "checkpoint_best_batch.pt"),
                    "checkpoint_path": str(estimator_checkpoint_path(runtime.spec.result_dir)),
                    "resource_snapshot": snapshot.as_dict(),
                },
            )
            if runtime not in active:
                active.append(runtime)
            self.counters.launched += 1
            # The copied controller records a generic gate-reopened event;
            # the original pause cause remains separately in launch context.
            self._record_event(runtime, "resume", reason="gate_reopened", snapshot=snapshot.as_dict())
        except Exception as exc:
            runtime.process = None
            runtime.pid = None
            runtime.status = RunStatus.PAUSED
            runtime.pause_reason = previous_reason
            runtime.exit_code = None
            runtime.last_exit_code = None
            runtime.ended_at = None
            runtime.last_error = "resume_launch_error"
            self.state_store.save(runtime)
            LOG.warning(
                "[RESUME] relaunch failed job=%s attempt=%s reason=%s error=%s; job remains paused",
                runtime.job_id,
                runtime.attempt,
                previous_reason,
                exc,
            )
            self._record_event(
                runtime, "resume_launch_error", reason=str(exc),
                snapshot=snapshot.as_dict(), log_event=False,
            )
            return False
        return True

    @staticmethod
    def _should_retry(settings: RunnerSettings, runtime: JobRuntime) -> bool:
        return settings.max_retries <= 0 or runtime.retry_count < settings.max_retries

    @staticmethod
    def _latest_attempt_text(path: str | None, fallback_bytes: int = 200_000) -> str:
        if not path:
            return ""
        marker = b"=== LAUNCH "
        try:
            with Path(path).open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                end = stream.tell()
                position = end
                carry = b""
                while position > 0:
                    size = min(64 * 1024, position)
                    position -= size
                    stream.seek(position)
                    combined = stream.read(size) + carry
                    index = combined.rfind(marker)
                    if index >= 0:
                        stream.seek(position + index)
                        return stream.read().decode("utf-8", errors="replace")
                    carry = combined[:len(marker) - 1]
                stream.seek(max(0, end - int(fallback_bytes)))
                return stream.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _classify_failure(self, runtime: JobRuntime, return_code: int) -> str:
        text = self._latest_attempt_text(runtime.log_path)
        lowered = text[-4000:].lower()
        full = text.lower()
        if "exact-resume failure:" in full or "lacks exact-resume guarantees" in full:
            return "inexact_checkpoint_resume"
        if any(marker in lowered for marker in (
            "cuda out of memory", "cudnn_status_alloc_failed", "cublas_status_alloc_failed", "cublascreate(handle)"
        )) or ("out of memory" in lowered and "cuda" in lowered):
            return "cuda_oom"
        if any(marker in lowered for marker in ("out of memory", "cannot allocate memory", "memoryerror")) or ("killed" in lowered and "oom" in lowered):
            return "memory_oom"
        return f"nonzero_exit:{return_code}"

    def _load_worker_result(self, runtime: JobRuntime) -> dict[str, Any] | None:
        path = Path(runtime.spec.result_dir) / "result.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _release_gate_locks(self, runtime: JobRuntime) -> None:
        """Release admission gates using the source's device-mode rule."""
        released: list[str] = []
        if runtime.device_mode == "gpu":
            released.extend(("gpu", "host"))
        elif runtime.device_mode == "cpu":
            released.append("host")
        if not released:
            return
        self.admission.release_for_device(runtime.device_mode)
        LOG.info(
            "[GATE] admission reopened job=%s released=%s",
            runtime.job_id,
            ",".join(released),
        )

    def _finalize(self, runtime: JobRuntime, *, success: bool, reason: str | None = None, snapshot: ResourceSnapshot | None = None) -> None:
        runtime.status = RunStatus.COMPLETED if success else RunStatus.FAILED
        runtime.ended_at = self.clock()
        if runtime.exit_code is not None:
            runtime.last_exit_code = runtime.exit_code
        runtime.last_error = reason
        if runtime.log_handle is not None:
            try:
                exit_code = runtime.exit_code
                if exit_code is None:
                    exit_code = runtime.last_exit_code
                runtime.log_handle.write(
                    f"=== FINISH {runtime.job_id} exit={exit_code} reason={reason or ''} ===\n"
                )
            except Exception:
                pass
            close_log_handle(runtime.log_handle)
            runtime.log_handle = None
        self.state_store.save(runtime)
        self.state_store.append_progress(
            runtime,
            "complete" if success else "fail",
            reason=reason,
            snapshot=snapshot.as_dict() if snapshot else None,
        )
        final_exit_code = runtime.exit_code if runtime.exit_code is not None else (runtime.last_exit_code or 0)
        self._record_event(
            runtime,
            "finalize",
            reason=reason,
            snapshot=snapshot.as_dict() if snapshot else None,
            exit_code=int(final_exit_code),
            log_event=False,
        )
        if success:
            self.counters.completed += 1
            result = self._load_worker_result(runtime)
            if result is not None:
                result_path = self.results_dir / "results.jsonl"
                with result_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, sort_keys=True, default=str) + "\n")
                    stream.flush()
            self._release_gate_locks(runtime)
            # The reference completion path re-enables GPU admission only
            # after a GPU child completes; CPU completion only re-enables CPU.
            if runtime.device_mode == "gpu":
                self.admission.gpu_launches_enabled = True
            self.admission.cpu_launches_enabled = True
            self.admission.pressure_backoff_pending = False
            self.admission.last_backoff_reason = None
            self.memory.values["last_backoff_ts"] = None
            self._save_controller_state()
        else:
            self.counters.terminal_failures += 1

    def _handle_exit(self, runtime: JobRuntime, active: list[JobRuntime], suspended: list[JobRuntime], queue: deque[JobRuntime], snapshot: ResourceSnapshot) -> None:
        if runtime.recovered_process:
            runtime.process = None
            runtime.recovered_process = False
            runtime.pid = None
            runtime.status = RunStatus.QUEUED
            runtime.exit_code = None
            runtime.ended_at = None
            runtime.last_error = "recovered_child_exit_status_unknown"
            self.counters.recovered_unknown_exit += 1
            if runtime in active:
                active.remove(runtime)
            queue.append(runtime)
            self.state_store.save(runtime)
            self.state_store.append_event(runtime, "recovered_child_exit_status_unknown", reason=runtime.last_error, snapshot=snapshot.as_dict())
            LOG.warning(
                "[RECOVER] child disappeared after orchestrator restart; requeued job=%s for checkpointed continuation",
                runtime.job_id,
            )
            return
        process = runtime.process
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            return
        runtime.exit_code = int(return_code)
        runtime.last_exit_code = int(return_code)
        runtime.ended_at = self.clock()
        if runtime in active:
            active.remove(runtime)
        if return_code == 0:
            self._finalize(runtime, success=True, snapshot=snapshot)
            return
        self.counters.failed_attempts += 1
        reason = self._classify_failure(runtime, int(return_code))
        runtime.last_error = reason
        runtime.process = None
        runtime.pid = None
        if reason == "cuda_oom" and runtime.device_mode == "gpu":
            runtime.gpu_pause_count += 1
            if snapshot.gpu is not None:
                self.memory.values["gpu_peak_vram_mib"] = snapshot.gpu.used_mib
            remaining_gpu = [candidate for candidate in active if candidate.live and candidate.is_gpu]
            if remaining_gpu:
                victim = sorted(remaining_gpu, key=self._architecture_priority, reverse=True)[0]
                self._pause(victim, active, suspended, "peer_cuda_oom", snapshot)
        if reason != "inexact_checkpoint_resume" and self._should_retry(self.settings, runtime):
            self._release_gate_locks(runtime)
            runtime.retry_count += 1
            self.counters.retries += 1
            runtime.status = RunStatus.RETRYING
            runtime.exit_code = None
            if runtime.log_handle is not None:
                try:
                    runtime.log_handle.write(
                        f"=== RETRY {runtime.job_id} exit={return_code} ===\n"
                    )
                    runtime.log_handle.flush()
                    runtime.log_handle.close()
                except Exception:
                    pass
                runtime.log_handle = None
            if reason in {"cuda_oom", "memory_oom"}:
                previous_scale = self.admission.batch_scale
                next_scale = max(1e-6, previous_scale * self.settings.batch_backoff_factor)
                if next_scale < previous_scale:
                    self.admission.batch_scale = next_scale
                    self.admission.pressure_backoff_pending = False
                    self.admission.last_backoff_reason = reason
                    self.memory.values["last_backoff_ts"] = time.strftime("%Y%m%d_%H%M%S")
                    self._save_controller_state()
            else:
                self.admission.pressure_backoff_pending = False
                self.admission.last_backoff_reason = None
                self.memory.values["last_backoff_ts"] = None
                self._save_controller_state()
            self.state_store.save(runtime)
            self._record_event(
                runtime, "retry", reason=reason,
                snapshot=snapshot.as_dict(), exit_code=int(return_code),
            )
            queue.append(runtime)
        else:
            self._finalize(runtime, success=False, reason=reason, snapshot=snapshot)
            # The source persists a terminal failure before releasing the
            # device-mode gate; retryable failures release before requeueing.
            self._release_gate_locks(runtime)

    @staticmethod
    def _architecture_priority(runtime: JobRuntime) -> tuple[int, int, str]:
        """Match the ledger's descending (depth, width, job id) peer order."""
        if runtime.spec.route.fixed_grid:
            widths = runtime.spec.route.fixed_grid[0]
            depth, width = len(widths), max(widths, default=0)
        else:
            settings = runtime.spec.worker_settings
            depth = int(runtime.spec.route.init_depth or settings.get("init_depth", 1))
            width = int(runtime.spec.route.init_width or settings.get("init_width", 1))
        return depth, width, runtime.job_id

    def _suspend_pressure_victim(self, active: list[JobRuntime], suspended: list[JobRuntime], snapshot: ResourceSnapshot) -> None:
        if not snapshot.gpu_pressure and not snapshot.host_pressure:
            return
        if snapshot.gpu_pressure:
            self.memory.values["gpu_peak_vram_mib"] = snapshot.gpu.used_mib if snapshot.gpu else 0.0
        if snapshot.host_pressure:
            self.memory.values["cpu_peak_host_mib"] = snapshot.host.total_mib - snapshot.host.available_mib
        pressure_kind = "gpu" if snapshot.gpu_pressure else "host"
        reasons = []
        if snapshot.gpu_pressure:
            reasons.append("gpu_memory_pressure")
        if snapshot.host_pressure:
            reasons.append("host_ram_pressure")
        reason = "+".join(reasons)
        candidates = [runtime for runtime in active if runtime.live and (pressure_kind != "gpu" or runtime.is_gpu)]
        samples = {
            runtime.job_id: self._sample_runtime_child(runtime)
            for runtime in candidates
        }
        if pressure_kind == "gpu":
            candidates.sort(key=lambda runtime: (
                samples[runtime.job_id].gpu_vram_mib,
                samples[runtime.job_id].host_rss_mib,
                self._architecture_priority(runtime),
            ), reverse=True)
        else:
            candidates.sort(key=lambda runtime: (
                samples[runtime.job_id].host_rss_mib,
                samples[runtime.job_id].gpu_vram_mib,
                self._architecture_priority(runtime),
            ), reverse=True)
        if candidates:
            if self._pause(candidates[0], active, suspended, reason, snapshot):
                # The source clears the transient backoff marker after a
                # successful hard-pressure pause. OOM peer pauses take a
                # separate path and intentionally do not do this.
                self.admission.pressure_backoff_pending = False
                self.admission.last_backoff_reason = None
                self.memory.values["last_backoff_ts"] = None
                LOG.warning(
                    "[PRESSURE] paused job=%s reason=%s active=%d suspended=%d host_avail=%s host_rss=%s gpu_free=%s gpu_used=%s",
                    candidates[0].job_id,
                    reason,
                    len(active),
                    len(suspended),
                    getattr(getattr(snapshot, "host", None), "available_mib", "-"),
                    getattr(snapshot, "active_rss_mib", "-"),
                    getattr(getattr(snapshot, "gpu", None), "free_mib", "-"),
                    getattr(getattr(snapshot, "gpu", None), "used_mib", "-"),
                )
        self._save_controller_state()

    def _install_signal_handlers(self, active: list[JobRuntime], suspended: list[JobRuntime]) -> None:
        def handler(signum, _frame):
            self.shutdown_signal = int(signum)
            self._shutdown(active, suspended)
            raise SystemExit(130)

        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM), getattr(signal, "SIGQUIT", signal.SIGTERM)):
            try:
                self._previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def _shutdown(self, active: list[JobRuntime], suspended: list[JobRuntime]) -> None:
        label = signal.Signals(self.shutdown_signal).name if self.shutdown_signal else "KeyboardInterrupt"
        LOG.warning(
            "[INTERRUPT] Shutdown requested by %s. Terminating active and suspended jobs...",
            label,
        )
        for runtime in [*active, *suspended]:
            if runtime.process is not None and runtime.live:
                try:
                    terminate_process_tree(runtime.process)
                except Exception:
                    pass
            if runtime.log_handle is not None:
                close_log_handle(runtime.log_handle)
            runtime.process = None
            runtime.pid = None
            runtime.exit_code = None
            runtime.last_exit_code = None
            runtime.ended_at = None
            runtime.status = RunStatus.PAUSED if runtime in suspended else RunStatus.QUEUED
            # Preserve a paused job's pressure cause so recovery still knows
            # which resource domain must reopen; persist interruption separately.
            if runtime not in suspended:
                # The source writes an empty string on active-to-queued
                # shutdown, reserving None for records without this field.
                runtime.pause_reason = ""
            runtime.last_error = f"interrupted:{label}"
            self.state_store.save(runtime)
        try:
            self._save_controller_state()
        except Exception:
            # The source treats controller/backoff persistence as best-effort
            # during shutdown, after per-job interruption state is written.
            pass

    def _write_final_reports(self) -> dict[str, Any]:
        screening_summary = None
        if self.settings.run_hardness_screen:
            screening_summary = build_screening_reports(
                self.plan.runs,
                self.results_dir,
                classification_ceiling=self.settings.classification_ceiling,
                regression_ceiling_r2=self.settings.regression_ceiling_r2,
                min_complexity_gain=self.settings.min_complexity_gain,
                min_hard_fold_fraction=self.settings.min_hard_fold_fraction,
                screen_seed_count=self.settings.screen_seed_count,
            )
        summary = {
            "run_root": str(self.run_root),
            "results_dir": str(self.results_dir),
            "run_signature": self.plan.run_signature,
            "manifest_jobs": len(self.plan.runs),
            "completed": self.counters.completed,
            "terminal_failures": self.counters.terminal_failures,
            "failed_attempts": self.counters.failed_attempts,
            "retries": self.counters.retries,
            "recovered_unknown_exit": self.counters.recovered_unknown_exit,
            "timestamp": utc_now(),
            "resource_snapshot": self.last_resource_snapshot,
            "hardness_screening": screening_summary,
            **summarize_live(self.run_root, self.plan.runs, self.state_store),
        }
        atomic_write_json(self.run_root / "final_report.json", summary)
        atomic_write_json(self.run_root / "suite_summary.json", summary)
        atomic_write_text(
            self.run_root / "final_report.md",
            _summary_markdown(summary, persisted_state=False, include_timestamp=True),
        )
        atomic_write_text(
            self.run_root / "suite_summary.md",
            _summary_markdown(summary, persisted_state=False, include_timestamp=False),
        )
        task_summary = self.run_root / "task_summary.csv"
        with task_summary.open("w", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("job_id", "status", "attempt", "retry_count", "exit_code", "last_exit_code", "result_dir"))
            for spec in self.plan.runs:
                record = self.state_store.load(spec.identity.run_id) or {}
                writer.writerow(record.get(key, "") for key in ("job_id", "status", "attempt", "retry_count", "exit_code", "last_exit_code", "result_dir"))
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass
        return summary

    def write_persisted_state_report(self) -> dict[str, Any]:
        """Regenerate summaries from job-state files without starting children.

        Invocation-local failure counters cannot be reconstructed exactly from
        state files, so this mode labels itself and leaves that counter unset.
        """
        statuses: dict[str, int] = {status.value: 0 for status in RunStatus}
        completed = terminal_failures = retries = jobs_with_state = 0
        records: list[dict[str, Any]] = []
        for spec in self.plan.runs:
            record = self.state_store.load(spec.identity.run_id) or {}
            if record:
                jobs_with_state += 1
            status = str(record.get("status", RunStatus.QUEUED.value)).lower()
            if status == RunStatus.LAUNCHING.value:
                status = RunStatus.RUNNING.value
            if status not in statuses:
                status = RunStatus.QUEUED.value
            statuses[status] += 1
            successful = status == RunStatus.COMPLETED.value and _coerce_int(record.get("exit_code"), -1) == 0
            completed += int(successful)
            terminal_failures += int(status == RunStatus.FAILED.value)
            retries += max(0, _coerce_int(record.get("retry_count"), 0))
            records.append(record)
        resource_snapshot = {}
        try:
            saved_status = json.loads((self.run_root / "status.json").read_text(encoding="utf-8"))
            if isinstance(saved_status, dict) and isinstance(saved_status.get("resource_snapshot"), dict):
                resource_snapshot = saved_status["resource_snapshot"]
        except Exception:
            pass
        summary = {
            "report_mode": "persisted_state_snapshot",
            "run_root": str(self.run_root),
            "results_dir": str(self.results_dir),
            "run_signature": self.plan.run_signature,
            "manifest_jobs": len(self.plan.runs),
            "jobs_with_state": jobs_with_state,
            "status_counts": statuses,
            "completed": completed,
            "terminal_failures": terminal_failures,
            "failed_attempts": None,
            "retries": retries,
            "recovered_unknown_exit": None,
            "timestamp": utc_now(),
            "resource_snapshot": resource_snapshot,
            **summarize_live(self.run_root, self.plan.runs, self.state_store),
        }
        atomic_write_json(self.run_root / "final_report.json", summary)
        atomic_write_json(self.run_root / "suite_summary.json", summary)
        atomic_write_text(
            self.run_root / "final_report.md",
            _summary_markdown(summary, persisted_state=True, include_timestamp=True),
        )
        atomic_write_text(
            self.run_root / "suite_summary.md",
            _summary_markdown(summary, persisted_state=True, include_timestamp=False),
        )
        task_summary = self.run_root / "task_summary.csv"
        with task_summary.open("w", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("job_id", "status", "attempt", "retry_count", "exit_code", "last_exit_code", "result_dir"))
            for spec, record in zip(self.plan.runs, records):
                writer.writerow((
                    spec.identity.run_id,
                    record.get("status", RunStatus.QUEUED.value),
                    record.get("attempt", ""),
                    record.get("retry_count", ""),
                    record.get("exit_code", ""),
                    record.get("last_exit_code", ""),
                    record.get("result_dir", spec.result_dir),
                ))
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass
        return summary

    def run(self) -> dict[str, Any]:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        seed_stats = seed_run_root_from_source(self.source_run_root, self.run_root)
        self.plan.write_manifest(self.run_root)
        atomic_write_json(self.run_root / "run_metadata.json", {
            "run_signature": self.plan.run_signature,
            "run_count": len(self.plan.runs),
            "seed_strategy": "sha256(base_seed, stable_run_identity)",
            "settings": self.settings.as_dict(),
            "source_seed_stats": seed_stats,
        })
        if self.settings.dry_run:
            atomic_write_json(self.run_root / "dry_run_jobs.json", [spec.as_dict() for spec in self.plan.runs])
            return {"dry_run": True, "run_signature": self.plan.run_signature, "job_count": len(self.plan.runs)}
        self._load_controller_state()
        active, suspended, queue = self._build_runtimes()
        self._install_signal_handlers(active, suspended)
        try:
            while queue or active or suspended:
                if self.shutdown_signal is not None:
                    self._shutdown(active, suspended)
                    raise SystemExit(130)
                self._status(active, suspended, queue)
                concurrency = self.settings.resolve_concurrency()
                snapshot, _latest_all, _latest_gpu = self._sample_snapshot(active, suspended, concurrency)
                counts = self._counts(active)
                current_host_used = snapshot.host.total_mib - snapshot.host.available_mib
                host_drop = float(self.memory.values.get("cpu_peak_host_mib", 0.0)) - current_host_used
                gpu_drop = float(self.memory.values.get("gpu_peak_vram_mib", 0.0)) - (snapshot.gpu.used_mib if snapshot.gpu else 0.0)
                for runtime in list(suspended):
                    if self._resume_ready(runtime, snapshot, host_drop, gpu_drop):
                        if self._resume(runtime, active, snapshot, concurrency):
                            suspended.remove(runtime)
                for runtime in list(active):
                    # The reference removes an active entry whose process
                    # handle disappeared without manufacturing an exit status.
                    if runtime.process is None:
                        if runtime in active:
                            active.remove(runtime)
                        continue
                    if not runtime.live:
                        self._handle_exit(runtime, active, suspended, queue, snapshot)
                self._suspend_pressure_victim(active, suspended, snapshot)
                counts = self._counts(active)
                # Fill the available queue slot until one child launches or no
                # route is currently admissible. Launch failures retry at the
                # queue tail and immediately continue this inner loop, matching
                # the source runner (including its uncapped-failure spin edge).
                successful_launch = False
                while queue and not successful_launch:
                    # The source resolves the live concurrency setting again
                    # for every queue attempt. A launch failure can change the
                    # control file before the next FIFO candidate is examined.
                    live_concurrency = self.settings.resolve_concurrency()
                    admission_budget = min(
                        scaled_launch_budget(
                            self.settings, live_concurrency, self.admission.batch_scale,
                        ),
                        snapshot.launch_capacity,
                    )
                    if counts.total >= admission_budget:
                        break
                    now = self.clock()
                    if now < self.admission.gpu_hold_until and now < self.admission.cpu_hold_until:
                        break
                    runtime = queue[0]
                    device = choose_device(
                        self.settings,
                        self.admission,
                        host=snapshot.host,
                        gpu=snapshot.gpu,
                        counts=counts,
                        now=now,
                    )
                    if device is None:
                        break
                    queue.popleft()
                    host_used_mib = snapshot.host.total_mib - snapshot.host.available_mib
                    latest_child_host_mib = float(
                        self.memory.values.get("last_child_host_mib", getattr(snapshot, "latest_child_rss_mib", 0.0))
                    )
                    latest_child_gpu_mib = float(
                        self.memory.values.get("last_child_gpu_mib", getattr(snapshot, "latest_child_vram_mib", 0.0))
                    )
                    latest_child_host_remaining = max(
                        0.0,
                        float(self.memory.values.get("last_host_remaining_mib", snapshot.host.available_mib - latest_child_host_mib)),
                    )
                    latest_child_gpu_remaining = (
                        max(
                            0.0,
                            float(self.memory.values.get("last_gpu_remaining_mib", snapshot.gpu.free_mib - latest_child_gpu_mib)),
                        )
                        if snapshot.gpu is not None else None
                    )
                    launch_context = {
                        "timestamp": utc_now(),
                        "host_total_mib": snapshot.host.total_mib,
                        "host_available_mib": snapshot.host.available_mib,
                        "host_available_effective_mib": snapshot.host.available_mib,
                        "host_used_effective_mib": host_used_mib,
                        "active_rss_mib": getattr(snapshot, "active_rss_mib", 0.0),
                        "gpu_total_mib": snapshot.gpu.total_mib if snapshot.gpu else None,
                        "gpu_free_mib": snapshot.gpu.free_mib if snapshot.gpu else None,
                        "gpu_free_effective_mib": snapshot.gpu.free_mib if snapshot.gpu else None,
                        "gpu_used_effective_mib": snapshot.gpu.used_mib if snapshot.gpu else None,
                        "active_count": counts.total,
                        "active_gpu_count": counts.gpu,
                        "active_cpu_count": counts.cpu,
                        # The reference stores the effective admission budget
                        # here: concurrency after batch scaling, capped by the
                        # resource-derived capacity. The uncapped capacity is
                        # still available as snapshot.launch_capacity.
                        "launch_capacity": admission_budget,
                        "resource_launch_capacity": snapshot.launch_capacity,
                        "batch_scale": self.admission.batch_scale,
                        "host_ram_stop_avail_mib": self.settings.host_ram_stop_avail_mib,
                        "host_ram_resume_avail_mib": self.settings.host_ram_resume_avail_mib,
                        "gpu_stop_avail_mib": self.settings.gpu_stop_avail_mib,
                        "gpu_resume_avail_mib": self.settings.gpu_resume_avail_mib,
                        "gpu_child_gate_close_buffer_mib": self.settings.child_gpu_gate_buffer_mib,
                        "host_child_gate_close_buffer_mib": self.settings.child_host_gate_buffer_mib,
                        "latest_child_job_id": self.memory.values.get("last_child_job_id"),
                        "latest_child_host_rss_mib": latest_child_host_mib,
                        "latest_child_gpu_vram_mib": latest_child_gpu_mib,
                        "latest_child_host_remaining_mib": latest_child_host_remaining,
                        "latest_child_gpu_remaining_mib": latest_child_gpu_remaining,
                        "host_gate_closed": self.admission.host_gate_closed,
                        "gpu_gate_closed": self.admission.gpu_gate_closed,
                        "gates": {
                            "host_closed": self.admission.host_gate_closed,
                            "gpu_closed": self.admission.gpu_gate_closed,
                        },
                    }
                    LOG.info(
                        "[RESOURCE] launch=%s queue=%d active=%d cap=%d host_avail=%d host_rss=%d gpu_free=%s gpu_used=%s last_child_rss=%s last_child_vram=%s gpu_hold=%.1fs cpu_hold=%.1fs",
                        runtime.job_id,
                        len(queue) + 1,
                        counts.total,
                        admission_budget,
                        int(snapshot.host.available_mib),
                        int(getattr(snapshot, "active_rss_mib", 0.0)),
                        int(snapshot.gpu.free_mib) if snapshot.gpu is not None else -1,
                        int(snapshot.gpu.used_mib) if snapshot.gpu is not None else -1,
                        int(latest_child_host_mib),
                        int(latest_child_gpu_mib),
                        max(0.0, self.admission.gpu_hold_until - now),
                        max(0.0, self.admission.cpu_hold_until - now),
                    )
                    try:
                        launch_job(
                            runtime,
                            settings=self.settings,
                            state_store=self.state_store,
                            repository_root=self.repository_root,
                            launch_budget=admission_budget,
                            device=device,
                            launch_context=launch_context,
                        )
                    except Exception as exc:
                        self.counters.failed_attempts += 1
                        runtime.last_error = f"launch_error:{type(exc).__name__}:{exc}"
                        LOG.exception("[LAUNCH] failed job=%s error=%s", runtime.job_id, exc)
                        if self._should_retry(self.settings, runtime):
                            runtime.last_exit_code = 1
                            runtime.retry_count += 1
                            runtime.status = RunStatus.RETRYING
                            runtime.exit_code = None
                            runtime.ended_at = None
                            self.counters.retries += 1
                            self.state_store.save(runtime)
                            queue.append(runtime)
                            self._record_event(runtime, "retry", reason="launch_error", exit_code=1)
                            continue
                        runtime.exit_code = 1
                        self._finalize(runtime, success=False, reason="launch_error", snapshot=snapshot)
                        continue
                    active.append(runtime)
                    counts = self._counts(active)
                    self.counters.launched += 1
                    self._record_event(runtime, "launch", reason=device, snapshot=snapshot.as_dict())
                    if device == "gpu":
                        self.admission.gpu_hold_until = now + self.settings.post_launch_sample_delay_seconds
                    else:
                        self.admission.cpu_hold_until = now + self.settings.post_launch_sample_delay_seconds
                    self._save_controller_state()
                    LOG.info(
                        "[HOLD] launched=%s device=%s waiting=%.1fs before next resource poll",
                        runtime.job_id,
                        device,
                        self.settings.post_launch_sample_delay_seconds,
                    )
                    if self.settings.post_launch_sample_delay_seconds > 0:
                        self.sleep(self.settings.post_launch_sample_delay_seconds)
                    self.admission.batch_scale = 1.0
                    self.admission.pressure_backoff_pending = False
                    self.admission.last_backoff_reason = None
                    self.memory.values["last_backoff_ts"] = None
                    self._save_controller_state()
                    successful_launch = True
                if queue or active or suspended:
                    self.sleep(max(0.1, self.settings.pressure_poll_seconds))
            # The source clears the logging-context lane counts before its
            # forced drain status line, then emits the final report bundle.
            self.logging_counts = LaneCounts(total=0, gpu=0, cpu=0)
            self._status(active, suspended, queue, force=True)
            summary = self._write_final_reports()
            return summary
        except KeyboardInterrupt:
            self._shutdown(active, suspended)
            raise SystemExit(130)
        finally:
            for sig, previous in self._previous_handlers.items():
                try:
                    signal.signal(sig, previous)
                except (ValueError, OSError):
                    pass
