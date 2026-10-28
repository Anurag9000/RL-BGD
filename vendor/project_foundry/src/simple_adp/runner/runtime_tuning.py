"""Best-effort CPU/thread/priority tuning ported into generic runner utilities."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .admission import parent_cpu_masked


def detected_cpus() -> tuple[int, ...]:
    try:
        cpus = tuple(sorted(os.sched_getaffinity(0)))
    except Exception:
        cpus = ()
    if not cpus:
        cpus = tuple(range(max(1, os.cpu_count() or 1)))
    frequencies: dict[int, int] = {}
    for cpu in cpus:
        candidates = (
            Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"),
            Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq"),
            Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/base_frequency"),
        )
        frequency = 0
        for candidate in candidates:
            try:
                frequency = int(candidate.read_text(encoding="utf-8").strip() or "0")
                break
            except Exception:
                continue
        frequencies[cpu] = frequency
    return tuple(sorted(cpus, key=lambda cpu: (-frequencies[cpu], cpu)))


def parse_cpu_list(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    return tuple(sorted(cpus))


def partition_cpus(cpus: tuple[int, ...], parts: int, slot: int) -> tuple[int, ...]:
    if not cpus:
        return ()
    parts = min(len(cpus), max(1, int(parts)))
    slot = min(parts - 1, max(0, int(slot)))
    base, remainder = divmod(len(cpus), parts)
    start = slot * base + min(slot, remainder)
    size = base + (1 if slot < remainder else 0)
    return cpus[start:start + size]


def stable_cpu_slot(job_key: str, parts: int) -> int:
    digest = hashlib.sha1(job_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(1, parts)


def current_concurrency_hint(explicit: int | None = None) -> int | None:
    # Match the source's positive-integer parser: an invalid/nonpositive
    # explicit hint is treated as absent, so the environment can still supply
    # a usable concurrency value.
    if explicit is not None:
        try:
            value = int(str(explicit))
        except Exception:
            value = None
        if value is not None and value > 0:
            return value
    raw = os.environ.get("TABULAR_CPU_JOB_CONCURRENCY")
    if raw is not None:
        try:
            value = int(raw)
        except Exception:
            value = None
        if value is not None and value > 0:
            return value
    return None


def resolve_thread_budget(core_count: int, hint: int | None) -> int:
    if hint is None or hint <= 1:
        return max(1, core_count)
    return max(1, core_count // hint)


def resolve_num_workers(_budget: int) -> int:
    return 0


def _apply_affinity(cpus: tuple[int, ...]) -> bool:
    if not cpus:
        return False
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpus))
            return True
        elif os.name == "nt":
            import ctypes

            mask = sum(1 << cpu for cpu in cpus if 0 <= cpu < 64)
            if mask:
                return bool(ctypes.windll.kernel32.SetProcessAffinityMask(
                    ctypes.windll.kernel32.GetCurrentProcess(), mask
                ))
    except Exception:
        pass
    return False


def _apply_affinity_from_env() -> None:
    """Apply the configured affinity or best-effort all online CPU IDs."""
    affinity_value = os.environ.get("TABULAR_CPU_AFFINITY_CPUS")
    try:
        configured_cpus = parse_cpu_list(affinity_value) if affinity_value else ()
    except ValueError:
        configured_cpus = ()
    if configured_cpus and _apply_affinity(configured_cpus):
        return
    online_cpus = tuple(range(max(1, os.cpu_count() or 1)))
    _apply_affinity(online_cpus)


def _apply_priority() -> None:
    if os.name == "nt":
        try:
            import ctypes

            kernel = ctypes.windll.kernel32
            priority_name = os.environ.get("TABULAR_WINDOWS_PRIORITY_CLASS", "high").strip().lower()
            priority_map = {
                "idle": 0x40, "below_normal": 0x4000, "normal": 0x20,
                "above_normal": 0x8000, "high": 0x80, "background": 0x00100000,
            }
            kernel.SetPriorityClass(kernel.GetCurrentProcess(), priority_map.get(priority_name, priority_map["high"]))
        except Exception:
            pass
        try:
            import ctypes

            kernel = ctypes.windll.kernel32
            class MemoryPriorityInformation(ctypes.Structure):
                _fields_ = [("MemoryPriority", ctypes.c_uint)]
            memory_name = os.environ.get("TABULAR_WINDOWS_MEMORY_PRIORITY", "normal").strip().lower()
            memory_map = {"very_low": 1, "low": 2, "medium": 3, "below_normal": 4, "normal": 5}
            info = MemoryPriorityInformation(memory_map.get(memory_name, 5))
            ctypes.windll.kernel32.SetProcessInformation(
                kernel.GetCurrentProcess(), 0, ctypes.byref(info), ctypes.sizeof(info)
            )
        except Exception:
            pass
        return
    try:
        os.nice(-20)
    except Exception:
        pass
    if shutil_which("ionice") is not None:
        try:
            subprocess.run(
                ["ionice", "-c2", "-n0", "-p", str(os.getpid())],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    sched_batch = getattr(os, "SCHED_BATCH", None)
    if sched_batch is not None:
        try:
            os.sched_setscheduler(0, sched_batch, os.sched_param(0))
        except Exception:
            pass


def _enforce_no_swap() -> None:
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6", use_errno=True).mlockall(3)  # MCL_CURRENT | MCL_FUTURE
        except Exception:
            pass
    elif os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetProcessWorkingSetSize(ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
        except Exception:
            pass


def launcher_child_env(
    base_env: dict[str, str],
    *,
    job_key: str,
    launch_budget: int | None = None,
    concurrency_hint: int | None = None,
    affinity_slot: int | None = None,
    shared_cpu: bool = False,
    cpu_only: bool = False,
    device_index: int = 0,
    device_mode: str | None = None,
) -> dict[str, str]:
    env = dict(base_env)
    truthy = {"1", "true", "yes", "on", "y"}
    inherited_visible = env.get("CUDA_VISIBLE_DEVICES")
    parent_cpu_only = parent_cpu_masked(env)
    if device_mode not in {None, "cpu", "gpu"}:
        raise ValueError(f"invalid child device mode: {device_mode!r}")
    if device_mode == "gpu" and (cpu_only or parent_cpu_only):
        raise RuntimeError("GPU child cannot override central CPU admission")
    cpu_only = bool(cpu_only or parent_cpu_only or device_mode == "cpu")
    cpus = detected_cpus()
    cores = len(cpus)
    explicit_hint = launch_budget if launch_budget is not None else concurrency_hint
    hint = current_concurrency_hint(explicit_hint)
    affinity_value = env.get("TABULAR_CPU_AFFINITY_CPUS") or os.environ.get(
        "TABULAR_CPU_AFFINITY_CPUS"
    )
    if affinity_value:
        try:
            parsed = parse_cpu_list(affinity_value)
            if parsed:
                cpus = parsed
        except ValueError:
            pass
    # Preserve source behavior: the inherited environment can opt into shared
    # mode even when the caller supplies the default/explicit False value.
    shared_cpu = bool(shared_cpu) or env.get("TABULAR_CHILD_SHARED_CPU", "").strip().lower() in truthy
    budget = max(1, cores) if shared_cpu else resolve_thread_budget(cores, hint)
    if shared_cpu:
        if cpus:
            env["GOMP_CPU_AFFINITY"] = ",".join(str(cpu) for cpu in cpus)
    elif affinity_slot is not None and hint and len(cpus) > 1:
        parts = min(max(1, int(hint)), len(cpus))
        cpus = partition_cpus(cpus, parts, affinity_slot)
    elif job_key and hint and len(cpus) > 1:
        parts = min(max(1, int(hint)), len(cpus))
        cpus = partition_cpus(cpus, parts, stable_cpu_slot(job_key, parts))
    if cpus:
        env["TABULAR_CPU_AFFINITY_CPUS"] = ",".join(str(cpu) for cpu in cpus)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(budget)
    env["TORCH_NUM_THREADS"] = str(budget)
    env["TORCH_INTEROP_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["OMP_WAIT_POLICY"] = "ACTIVE"
    env["TABULAR_CPU_THREADS"] = str(budget)
    env["TABULAR_CPU_WORKERS"] = "0"
    env["TABULAR_CPU_CORES"] = str(cores)
    env["TABULAR_CPU_JOB_CONCURRENCY"] = "1" if shared_cpu else str(max(1, int(hint or 1)))
    if shared_cpu:
        env["OMP_PROC_BIND"] = "spread"
        env["OMP_PLACES"] = "cores"
        env["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    inherited = env.get("CUDA_VISIBLE_DEVICES", "")
    if cpu_only:
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            env[key] = ""
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["SIMPLE_ADP_DISABLE_GPU_ACCELERATORS"] = "1"
        env["TRAINING_CONTROL_CPU_ONLY"] = "1"
        env["TABULAR_BACKEND"] = "cpu"
    else:
        if inherited.strip() and inherited.strip() not in {"-1"}:
            values = [value.strip() for value in inherited.split(",")]
            if len(values) == 1:
                assigned = values[0]
            elif str(device_index) in values:
                assigned = str(device_index)
            else:
                raise ValueError("GPU index is not in the inherited CUDA device mask")
        else:
            assigned = str(device_index)
        env["CUDA_VISIBLE_DEVICES"] = assigned
        # Keep the backend-admission flag explicit, matching the source
        # child's `...DISABLE_GPU_ACCELERATORS=0` setting.
        env["SIMPLE_ADP_DISABLE_GPU_ACCELERATORS"] = "0"
        env.pop("TRAINING_CONTROL_CPU_ONLY", None)
        env["TABULAR_BACKEND"] = "gpu"
        env["TABULAR_LOCAL_GPU_INDEX"] = "0"
    return env


@dataclass(frozen=True, slots=True)
class RuntimeBootstrap:
    label: str
    cores: int
    workers: int
    torch_threads: int
    torch_interop_threads: int = 1


def maybe_reexec_systemd_scope(argv: list[str] | None = None, *, label: str = "runner") -> bool:
    """Best-effort user scope re-exec; return false when skipped or exec fails."""
    if os.name != "posix" or os.environ.get("TABULAR_SKIP_SYSTEMD_SCOPE") == "1" or os.environ.get("TABULAR_SYSTEMD_SCOPED") == "1":
        return False
    if argv is not None:
        arguments = list(argv)
    else:
        # sys.argv[0] becomes the module's source path for ``python -m``.
        # Re-executing that path as a script breaks package-relative imports,
        # so preserve the original interpreter flags and module invocation.
        original = getattr(sys, "orig_argv", None)
        if original and Path(original[0]).resolve() == Path(sys.executable).resolve():
            arguments = list(original[1:])
        else:
            arguments = list(sys.argv)
    if arguments[:1] and arguments[0] in {"-c", "-"}:
        return False
    executable = shutil_which("systemd-run")
    if executable is None:
        return False
    cpus = detected_cpus()
    properties = [
        "CPUWeight=10000", "StartupCPUWeight=10000", "IOWeight=10000", "StartupIOWeight=10000",
        "MemorySwapMax=0", "TasksMax=infinity", "CPUAccounting=yes", "IOAccounting=yes",
        "MemoryAccounting=yes", "TasksAccounting=yes",
    ]
    try:
        systemctl = shutil_which("systemctl")
        result = subprocess.run(
            [systemctl, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ) if systemctl else None
        first_line = (result.stdout or "").splitlines()[0] if result is not None and result.stdout else ""
        version_fields = first_line.split()
        if len(version_fields) >= 2 and version_fields[0].lower() == "systemd":
            if int(version_fields[1]) >= 253:
                properties.append("MemoryZSwapMax=0")
    except Exception:
        pass
    if cpus:
        properties.append("AllowedCPUs=" + ",".join(str(cpu) for cpu in cpus))
    command = ["systemd-run", "--user", "--scope", "--quiet", "--same-dir", "--collect", "--slice=app-mlps-training.slice", f"--description=tabular:{label}"]
    for prop in properties:
        command.extend(["-p", prop])
    env = dict(os.environ)
    env["TABULAR_SYSTEMD_SCOPED"] = "1"
    try:
        os.execvpe("systemd-run", [*command, sys.executable, *arguments], env)
    except Exception:
        return False
    return True


def shutil_which(executable: str) -> str | None:
    from shutil import which

    return which(executable)


def bootstrap_runtime(label: str = "simple-adp-suite", *, apply_os_tuning: bool = True) -> RuntimeBootstrap:
    maybe_reexec_systemd_scope(label=label)
    _apply_affinity_from_env()
    cpus = detected_cpus()
    hint = current_concurrency_hint()
    threads = resolve_thread_budget(len(cpus), hint)
    workers = resolve_num_workers(threads)
    env_updates = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "GOTO_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "TORCH_NUM_THREADS": str(threads),
        "TORCH_INTEROP_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "OMP_WAIT_POLICY": "ACTIVE",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
        "KMP_AFFINITY": "granularity=fine,scatter",
        "TABULAR_CHILD_SHARED_CPU": os.environ.get("TABULAR_CHILD_SHARED_CPU", "0"),
        "TABULAR_CPU_THREADS": str(threads),
        "TABULAR_CPU_WORKERS": "0",
        "TABULAR_CPU_CORES": str(len(cpus)),
    }
    os.environ.update(env_updates)
    if apply_os_tuning:
        _apply_priority()
        _enforce_no_swap()
    try:
        import torch

    except ImportError:
        torch = None
    if torch is not None:
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    return RuntimeBootstrap(label, len(cpus), workers, threads)
