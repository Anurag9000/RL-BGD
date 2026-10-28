"""Host, GPU, child-tree, and non-Python external-memory observations."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Protocol, runtime_checkable

import psutil

MIB = 1024.0 * 1024.0


@dataclass(frozen=True, slots=True)
class HostMemorySample:
    total_mib: float
    available_mib: float
    used_percent: float
    swap_total_mib: float
    swap_used_mib: float
    swap_percent: float


@dataclass(frozen=True, slots=True)
class GPUMemorySample:
    total_mib: float
    used_mib: float
    free_mib: float
    used_percent: float


@dataclass(frozen=True, slots=True)
class ProcessMemorySample:
    host_rss_mib: float = 0.0
    gpu_vram_mib: float = 0.0
    pid_set: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    host: HostMemorySample
    gpu: GPUMemorySample | None
    active_count: int
    active_gpu_count: int
    active_cpu_count: int
    active_rss_mib: float
    latest_child_rss_mib: float
    latest_child_vram_mib: float
    external_host_mib: float
    external_gpu_mib: float
    external_host_pids: frozenset[int]
    external_gpu_pids: frozenset[int]
    host_pressure: bool
    gpu_pressure: bool
    launch_capacity: int

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["external_host_pids"] = sorted(self.external_host_pids)
        result["external_gpu_pids"] = sorted(self.external_gpu_pids)
        return result


@runtime_checkable
class ResourceProvider(Protocol):
    """Injectable source for the coordinator's host, GPU, and process samples."""

    def sample_host(self) -> HostMemorySample: ...
    def sample_gpu(self, device_index: int) -> GPUMemorySample | None: ...
    def process_tree(self, pid: int) -> frozenset[int]: ...
    def sample_active_rss(self, pids: Iterable[int]) -> float: ...
    def sample_child(self, pid: int | None, *, gpu_device_index: int, gpu_mode: bool) -> ProcessMemorySample: ...
    def sample_external_host(self, excluded_pids: Iterable[int]) -> ProcessMemorySample: ...
    def sample_external_gpu(self, device_index: int, excluded_pids: Iterable[int]) -> ProcessMemorySample: ...


@dataclass(slots=True)
class CallbackResourceProvider:
    """Default provider assembled from system samplers, replaceable in tests/plugins."""

    host_sampler: Callable[[], HostMemorySample]
    gpu_sampler: Callable[[int], GPUMemorySample | None]
    tree_sampler: Callable[[int], frozenset[int]]
    child_sampler: Callable[..., ProcessMemorySample]
    external_host_sampler: Callable[[Iterable[int]], ProcessMemorySample]
    external_gpu_sampler: Callable[[int, Iterable[int]], ProcessMemorySample]

    def sample_host(self) -> HostMemorySample:
        return self.host_sampler()

    def sample_gpu(self, device_index: int) -> GPUMemorySample | None:
        return self.gpu_sampler(device_index)

    def process_tree(self, pid: int) -> frozenset[int]:
        return self.tree_sampler(pid)

    def sample_active_rss(self, pids: Iterable[int]) -> float:
        return sample_active_rss_mib(pids)

    def sample_child(self, pid: int | None, *, gpu_device_index: int, gpu_mode: bool) -> ProcessMemorySample:
        return self.child_sampler(pid, gpu_device_index=gpu_device_index, gpu_mode=gpu_mode)

    def sample_external_host(self, excluded_pids: Iterable[int]) -> ProcessMemorySample:
        return self.external_host_sampler(excluded_pids)

    def sample_external_gpu(self, device_index: int, excluded_pids: Iterable[int]) -> ProcessMemorySample:
        return self.external_gpu_sampler(device_index, excluded_pids)


def sample_host_memory() -> HostMemorySample:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    # The source converts OS byte counters to whole MiB with floor division;
    # keeping fractional MiB here can move strict pressure/gate boundaries.
    total = int(virtual.total // MIB)
    available = int(virtual.available // MIB)
    return HostMemorySample(
        total_mib=total,
        available_mib=available,
        used_percent=float(virtual.percent),
        swap_total_mib=int(swap.total // MIB),
        swap_used_mib=int(swap.used // MIB),
        swap_percent=float(swap.percent),
    )


def sample_gpu_memory(device_index: int) -> GPUMemorySample | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi", f"--id={int(device_index)}",
                "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        row = output.decode("utf-8", errors="ignore").strip().splitlines()[0]
        total_raw, used_raw = row.split(",", 1)
        total, used = int(float(total_raw.strip())), int(float(used_raw.strip()))
        free = max(0, total - used)
        used_percent = (float(used) / float(total)) * 100.0 if total > 0 else 0.0
        return GPUMemorySample(total, used, free, used_percent)
    except Exception:
        return None


def process_tree_pids(pid: int) -> frozenset[int]:
    root_pid = int(pid)
    pids = {root_pid}
    try:
        process = psutil.Process(root_pid)
        for child in process.children(recursive=True):
            try:
                pids.add(int(child.pid))
            except Exception:
                continue
    except Exception:
        # The source helper starts with the known root PID and keeps it even
        # when psutil cannot inspect that process or enumerate its children.
        pass
    return frozenset(pids)


def sample_active_rss_mib(pids: Iterable[int]) -> float:
    """Sum active-tree RSS; one failed lookup makes the source sum zero."""
    try:
        total = sum(psutil.Process(int(pid)).memory_info().rss for pid in pids)
    except Exception:
        return 0.0
    return float(total) / MIB


def _gpu_process_rows(device_index: int) -> list[tuple[int, str, float]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi", f"--id={int(device_index)}",
                "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        rows: list[tuple[int, str, float]] = []
        for line in output.decode("utf-8", errors="ignore").strip().splitlines():
            # Match the source's CSV handling: consume the first three cells
            # and tolerate extra comma-delimited cells after used_memory.
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                rows.append((int(parts[0]), parts[1], float(parts[2])))
            except Exception:
                continue
        return rows
    except Exception:
        return []


def sample_child_memory(pid: int | None, *, gpu_device_index: int, gpu_mode: bool) -> ProcessMemorySample:
    if pid is None:
        return ProcessMemorySample()
    pids = process_tree_pids(pid)
    rss_bytes = 0
    for child_pid in pids:
        try:
            rss_bytes += psutil.Process(child_pid).memory_info().rss
        except Exception:
            continue
    gpu_mib = 0.0
    if gpu_mode and pids:
        try:
            gpu_mib = sum(memory for proc_pid, _name, memory in _gpu_process_rows(gpu_device_index) if proc_pid in pids)
        except (OSError, subprocess.SubprocessError):
            gpu_mib = 0.0
    return ProcessMemorySample(rss_bytes / MIB, gpu_mib, pids)


def _is_python_process_name(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    return "python" in normalized or normalized.startswith("py")


def sample_external_host_memory(excluded_pids: Iterable[int]) -> ProcessMemorySample:
    excluded = set(excluded_pids)
    total_rss = 0
    pids: set[int] = set()
    for process in psutil.process_iter(attrs=["pid", "name", "cmdline", "memory_info"]):
        try:
            info = process.info
            pid = int(info.get("pid") or process.pid)
            if pid in excluded:
                continue
            name = str(info.get("name") or "")
            command = " ".join(info.get("cmdline") or [])
            if _is_python_process_name(name) or _is_python_process_name(command):
                continue
            memory_info = info.get("memory_info")
            info = memory_info if memory_info is not None else process.memory_info()
            total_rss += info.rss
            pids.add(pid)
        except Exception:
            continue
    return ProcessMemorySample(host_rss_mib=total_rss / MIB, pid_set=frozenset(pids))


def sample_external_gpu_memory(device_index: int, excluded_pids: Iterable[int]) -> ProcessMemorySample:
    excluded = set(excluded_pids)
    try:
        rows = _gpu_process_rows(device_index)
    except (OSError, subprocess.SubprocessError):
        return ProcessMemorySample()
    pids: set[int] = set()
    total = 0.0
    for pid, name, memory in rows:
        if pid in excluded or _is_python_process_name(name):
            continue
        pids.add(pid)
        total += memory
    return ProcessMemorySample(gpu_vram_mib=total, pid_set=frozenset(pids))


def external_drop_reopened(
    *,
    previous_mib: float,
    current_mib: float,
    previous_pids: Iterable[int],
    current_pids: Iterable[int],
    threshold_mib: float,
) -> bool:
    return (
        previous_mib > 0
        and previous_mib - current_mib >= threshold_mib
        and bool(set(previous_pids) - set(current_pids))
    )
