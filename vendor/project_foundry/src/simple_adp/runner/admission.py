"""Admission capacity, lane gates, holds, and pressure-resume predicates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import os
from typing import Literal

from .config import RunnerSettings, UNBOUNDED_CONCURRENCY
from .resources import GPUMemorySample, HostMemorySample

DeviceMode = Literal["gpu", "cpu"]


def parent_cpu_masked(env: dict[str, str] | None = None) -> bool:
    """Return whether the launcher environment explicitly admits CPU only."""
    values = os.environ if env is None else env
    truthy = {"1", "true", "yes", "on", "y"}
    if any(values.get(key, "").strip().lower() == "cpu" for key in ("TRAINING_CONTROL_BACKEND", "TABULAR_BACKEND")):
        return True
    for key in (
        "CPU_ONLY", "TABULAR_CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
        "OPF_ADP_DISABLE_GPU_ACCELERATORS", "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS",
    ):
        if values.get(key, "").strip().lower() in truthy:
            return True
    visible = values.get("CUDA_VISIBLE_DEVICES")
    return visible is not None and visible.strip() in {"", "-1"}


@dataclass(slots=True)
class AdmissionState:
    gpu_gate_closed: bool = False
    host_gate_closed: bool = False
    gpu_gate_lock_job_id: str | None = None
    host_gate_lock_job_id: str | None = None
    gpu_launches_enabled: bool = True
    cpu_launches_enabled: bool = True
    gpu_hold_until: float = 0.0
    cpu_hold_until: float = 0.0
    batch_scale: float = 1.0
    pressure_backoff_pending: bool = False
    last_backoff_reason: str | None = None

    def invocation_reset(self, settings: RunnerSettings) -> None:
        self.gpu_gate_closed = self.host_gate_closed = False
        self.gpu_gate_lock_job_id = self.host_gate_lock_job_id = None
        self.gpu_launches_enabled = self.cpu_launches_enabled = True
        self.gpu_hold_until = self.cpu_hold_until = 0.0
        self.batch_scale = 1.0
        self.pressure_backoff_pending = False
        self.last_backoff_reason = None

    def release_for_device(self, device: DeviceMode) -> None:
        # The source helper's ownership is device-mode based, not lock-ID based.
        if device == "gpu":
            self.gpu_gate_closed = False
            self.host_gate_closed = False
            self.gpu_gate_lock_job_id = None
            self.host_gate_lock_job_id = None
            self.gpu_launches_enabled = True
            self.cpu_launches_enabled = True
        else:
            self.host_gate_closed = False
            self.host_gate_lock_job_id = None
            self.cpu_launches_enabled = True


@dataclass(frozen=True, slots=True)
class LaneCounts:
    total: int
    gpu: int
    cpu: int


def resolve_launch_capacity(
    settings: RunnerSettings,
    *,
    concurrency: int,
    host_available_mib: float,
    gpu: GPUMemorySample | None,
    active_count: int,
) -> int:
    if settings.launch_host_reserve_mib <= 0:
        host_slots = UNBOUNDED_CONCURRENCY
    else:
        # Preserve the source's float floor-division semantics at exact and
        # near-exact reserve boundaries (which can differ from floor(a / b)).
        host_slots = max(1, int(max(0.0, host_available_mib) // settings.launch_host_reserve_mib))
    if settings.launch_gpu_reserve_mib <= 0:
        gpu_slots = UNBOUNDED_CONCURRENCY
    elif gpu is None:
        gpu_slots = host_slots
    else:
        gpu_slots = max(1, int(max(0.0, gpu.free_mib) // settings.launch_gpu_reserve_mib))
    capacity = max(1, min(concurrency, host_slots, gpu_slots))
    if settings.max_active_jobs > 0:
        capacity = min(capacity, settings.max_active_jobs)
    capacity = max(1, capacity)
    if active_count and concurrency < UNBOUNDED_CONCURRENCY:
        capacity = min(capacity, active_count + 1)
    return max(1, capacity)


def launch_budget(settings: RunnerSettings, concurrency: int, batch_scale: float) -> int:
    return max(1, math.ceil(concurrency * batch_scale))


def hard_pressure(settings: RunnerSettings, host: HostMemorySample, gpu: GPUMemorySample | None) -> tuple[bool, bool]:
    host_pressure = host.available_mib < settings.host_ram_stop_avail_mib
    gpu_pressure = gpu is not None and gpu.free_mib < settings.gpu_stop_avail_mib
    return host_pressure, gpu_pressure


def update_soft_gates(
    state: AdmissionState,
    *,
    settings: RunnerSettings,
    host_pressure: bool,
    gpu_pressure: bool,
    gpu_sample_present: bool,
    latest_gpu_job: str | None,
    latest_host_job: str | None,
    latest_child_gpu_mib: float,
    latest_child_host_mib: float,
    host_available_mib: float,
    gpu_free_mib: float | None,
) -> None:
    if not state.gpu_gate_closed and latest_gpu_job is not None and gpu_sample_present:
        if gpu_pressure or latest_child_gpu_mib + settings.child_gpu_gate_buffer_mib > float(gpu_free_mib or 0.0):
            state.gpu_gate_closed = True
            state.gpu_gate_lock_job_id = latest_gpu_job
            state.gpu_launches_enabled = False
    if not state.host_gate_closed and latest_host_job is not None:
        if host_pressure or latest_child_host_mib + settings.child_host_gate_buffer_mib > host_available_mib:
            state.host_gate_closed = True
            state.host_gate_lock_job_id = latest_host_job
            state.cpu_launches_enabled = False


def reopen_external_gates(state: AdmissionState, *, host_drop: bool, gpu_drop: bool) -> None:
    if gpu_drop and state.gpu_gate_closed:
        state.gpu_gate_closed = False
        state.gpu_gate_lock_job_id = None
        state.gpu_launches_enabled = True
    if host_drop and state.host_gate_closed:
        state.host_gate_closed = False
        state.host_gate_lock_job_id = None
        state.cpu_launches_enabled = True


def choose_device(
    settings: RunnerSettings,
    state: AdmissionState,
    *,
    host: HostMemorySample,
    gpu: GPUMemorySample | None,
    counts: LaneCounts,
    now: float,
) -> DeviceMode | None:
    finite = settings.effective_concurrency < UNBOUNDED_CONCURRENCY
    total_cap_full = settings.max_active_jobs > 0 and counts.total >= settings.max_active_jobs
    gpu_cap_full = settings.max_active_gpu_jobs > 0 and counts.gpu >= settings.max_active_gpu_jobs
    gpu_intended = (
        not settings.cpu_only
        and state.gpu_launches_enabled
        and not state.gpu_gate_closed
        and not state.host_gate_closed
        and host.available_mib >= settings.host_ram_resume_avail_mib
        and gpu is not None
        and gpu.free_mib >= settings.gpu_resume_avail_mib
        and not gpu_cap_full
        and now >= state.gpu_hold_until
    )
    host_base_intended = (
        state.cpu_launches_enabled
        and not state.host_gate_closed
        and host.available_mib >= settings.host_ram_resume_avail_mib
        and now >= state.cpu_hold_until
    )
    host_intended = host_base_intended and not total_cap_full
    if settings.cpu_only or parent_cpu_masked():
        # The reference's CPU-only/parent-mask branch returns before the
        # selector's total-cap predicate. The controller checks its launch
        # budget before calling this helper, so keep that source quirk here.
        return "cpu" if host_base_intended else None
    # The source's fixed and gpu_first labels allow GPU whenever its own
    # admission predicate passes. pressure_aware also requires the CPU lane
    # predicate, including its hold, before admitting a GPU launch.
    if settings.scheduler in {"fixed", "gpu_first"} and gpu_intended:
        return "gpu"
    if settings.scheduler == "pressure_aware" and gpu_intended and host_intended:
        return "gpu"
    if host_intended:
        return "cpu"
    return None


def pressure_pause_reopened(
    *,
    pause_reason: str,
    device: DeviceMode,
    host_available_mib: float,
    gpu_free_mib: float | None,
    host_used_mib: float,
    gpu_used_mib: float | None,
    settings: RunnerSettings,
    host_peak_used_mib: float,
    gpu_peak_used_mib: float,
    external_host_drop_mib: float,
    external_gpu_drop_mib: float,
) -> bool:
    reason = str(pause_reason or "").lower()
    host_pause = "host" in reason or "swap" in reason or device == "cpu"
    gpu_pause = "gpu" in reason or "cuda" in reason
    if device == "gpu" and not (host_pause or gpu_pause):
        gpu_pause = True
    threshold = settings.external_memory_drop_mib
    host_ready = (
        host_available_mib >= settings.host_ram_resume_avail_mib
        or (host_peak_used_mib > 0 and host_peak_used_mib - host_used_mib >= threshold)
        or external_host_drop_mib >= threshold
    )
    gpu_ready = gpu_free_mib is not None and (
        gpu_free_mib >= settings.gpu_resume_avail_mib
        or (gpu_used_mib is not None and gpu_peak_used_mib > 0 and gpu_peak_used_mib - gpu_used_mib >= threshold)
        or external_gpu_drop_mib >= threshold
    )
    return (not host_pause or host_ready) and (not gpu_pause or gpu_ready)
