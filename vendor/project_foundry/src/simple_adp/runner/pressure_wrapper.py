"""Standalone percent-pressure wrapper with OS pause/continue semantics."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import psutil

from .config import StandalonePressureSettings
from .processes import terminate_standalone_pressure_tree
from .resources import MIB


@dataclass(frozen=True, slots=True)
class PressureSample:
    host_percent: float
    host_used_mib: float
    swap_percent: float
    gpu_percent: float
    gpu_used_mib: float
    python_host_mib: float
    python_gpu_mib: float

    @property
    def external_host_mib(self) -> float:
        return self.host_used_mib - self.python_host_mib

    @property
    def external_gpu_mib(self) -> float:
        return self.gpu_used_mib - self.python_gpu_mib


def _gpu_pressure_sample(device_index: int = 0) -> tuple[float, int]:
    total_mib = 0
    used_mib = 0
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        row = output.decode("utf-8").strip().splitlines()[0]
        total_text, used_text = (part.strip() for part in row.split(",", 1))
        total_mib = int(float(total_text))
        used_mib = int(float(used_text))
    except Exception:
        # Source behavior treats every query or parse failure as zero pressure.
        pass
    if total_mib <= 0:
        return 0.0, 0
    percent = max(0.0, min(100.0, float(used_mib) / float(total_mib) * 100.0))
    return percent, used_mib


def _sample_host_gpu_pressure(device_index: int = 0) -> PressureSample:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu_percent, gpu_used = _gpu_pressure_sample(device_index)
    return PressureSample(
        host_percent=float(virtual.percent),
        host_used_mib=float(virtual.used) / MIB,
        swap_percent=float(swap.percent),
        gpu_percent=gpu_percent,
        gpu_used_mib=gpu_used,
        python_host_mib=0.0,
        python_gpu_mib=0.0,
    )


def _sample_python_usage(device_index: int = 0) -> tuple[float, float]:
    python_host = 0.0
    for process in psutil.process_iter(["name", "memory_info"]):
        try:
            name = process.info["name"]
            if name and ("python" in name.lower() or "python3" in name.lower()):
                python_host += process.info["memory_info"].rss / MIB
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            TypeError,
            KeyError,
        ):
            continue
    python_gpu = 0.0
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        for line in output.decode("utf-8").strip().splitlines():
            pieces = [part.strip() for part in line.split(",")]
            if len(pieces) < 2:
                continue
            name, memory_text = pieces[0].lower(), pieces[1]
            if memory_text.isdigit() and ("python" in name or "python3" in name):
                python_gpu += float(memory_text)
    except Exception:
        pass
    return python_host, python_gpu


def _with_python_usage(sample: PressureSample, device_index: int) -> PressureSample:
    python_host, python_gpu = _sample_python_usage(device_index)
    return PressureSample(
        host_percent=sample.host_percent,
        host_used_mib=sample.host_used_mib,
        swap_percent=sample.swap_percent,
        gpu_percent=sample.gpu_percent,
        gpu_used_mib=sample.gpu_used_mib,
        python_host_mib=python_host,
        python_gpu_mib=python_gpu,
    )


def _external_drop_reopens_gates(
    *,
    host_ready: bool,
    gpu_ready: bool,
    host_drop_mib: float,
    gpu_drop_mib: float,
    threshold_mib: float,
) -> bool:
    """Apply the wrapper's independent host and GPU resume conditions."""
    drop_enabled = threshold_mib > 0.0
    host_gate_ready = host_ready or (drop_enabled and host_drop_mib >= threshold_mib)
    gpu_gate_ready = gpu_ready or (drop_enabled and gpu_drop_mib >= threshold_mib)
    return host_gate_ready and gpu_gate_ready


def sample_pressure(device_index: int = 0) -> PressureSample:
    """Return a full pressure/external-use sample for direct callers."""
    return _with_python_usage(_sample_host_gpu_pressure(device_index), device_index)


def _pause_child(process: subprocess.Popen, paused: bool) -> None:
    # The child may exit after the outer polling condition but before this
    # signal. Match the source helper's own liveness check at that boundary.
    if process.poll() is not None:
        return
    if os.name == "posix":
        process_group = os.getpgid(process.pid)
        os.killpg(process_group, signal.SIGCONT if not paused else signal.SIGSTOP)
        return
    try:
        root = psutil.Process(process.pid)
        targets = [*root.children(recursive=True), root]
    except psutil.Error:
        targets = []
    for target in targets:
        try:
            target.resume() if not paused else target.suspend()
        except psutil.Error:
            pass


class PressureAwareWrapper:
    def __init__(self, command: list[str], settings: StandalonePressureSettings, *, gpu_device_index: int = 0, sleeper=time.sleep):
        if not command:
            raise ValueError("a child command is required after --")
        self.command = command
        self.settings = settings
        self.gpu_device_index = gpu_device_index
        self.sleep = sleeper
        self.process: subprocess.Popen | None = None
        self.paused = False
        self.host_peak_external = 0.0
        self.gpu_peak_external = 0.0

    def _interrupt(self, signum, _frame) -> None:
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(f"\n[INTERRUPT] Caught {signame}. Terminating active child tree...", flush=True)
        if self.process is not None:
            terminate_standalone_pressure_tree(self.process)
        raise SystemExit(130)

    def run(self) -> int:
        print(f"[PRESSURE AWARE] Launching: {' '.join(self.command)}", flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
            try:
                signal.signal(sig, self._interrupt)
            except Exception:
                pass
        try:
            while True:
                if self.paused:
                    # The reference's paused branch deliberately does not poll
                    # or look up the child PID while waiting for both gates.
                    sample = _with_python_usage(
                        _sample_host_gpu_pressure(self.gpu_device_index),
                        self.gpu_device_index,
                    )
                    self.host_peak_external = max(self.host_peak_external, sample.external_host_mib)
                    self.gpu_peak_external = max(self.gpu_peak_external, sample.external_gpu_mib)
                    host_ready = sample.host_percent <= self.settings.host_resume_pct
                    gpu_ready = sample.gpu_percent <= self.settings.gpu_resume_pct
                    host_drop = self.host_peak_external - sample.external_host_mib
                    gpu_drop = self.gpu_peak_external - sample.external_gpu_mib
                    if _external_drop_reopens_gates(
                        host_ready=host_ready,
                        gpu_ready=gpu_ready,
                        host_drop_mib=host_drop,
                        gpu_drop_mib=gpu_drop,
                        threshold_mib=self.settings.external_memory_drop_mib,
                    ):
                        # Resume is a no-op if an external actor ended the
                        # stopped child; after gates reopen the source outer
                        # loop starts a replacement command in that case.
                        if self.process is not None and self.process.poll() is None:
                            _pause_child(self.process, False)
                        else:
                            self.process = None
                        print(
                            f"\n[STATE] admission gate reopened by RAM/GPU drop or thresholds met "
                            f"host_drop={host_drop:.1f} gpu_drop={gpu_drop:.1f} "
                            f"host={sample.host_percent:.2f} swap={sample.swap_percent:.2f} "
                            f"gpu={sample.gpu_percent:.2f}",
                            flush=True,
                        )
                        self.paused = False
                        self.host_peak_external = 0.0
                        self.gpu_peak_external = 0.0
                        continue
                    self.sleep(max(0.1, self.settings.poll_seconds))
                    continue

                # The source checks for a missing/exited handle before launch,
                # then checks the child again in the live polling condition.
                if self.process is None or self.process.poll() is not None:
                    self.process = subprocess.Popen(
                        self.command,
                        start_new_session=(os.name == "posix"),
                        creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0),
                    )
                try:
                    psutil.Process(self.process.pid)
                except psutil.NoSuchProcess:
                    return int(self.process.wait())

                # This inner poll reproduces the source's second child check
                # after a fresh Popen and its per-iteration live check.
                while self.process.poll() is None:
                    sample = _sample_host_gpu_pressure(self.gpu_device_index)
                    pressure_requested = (
                        sample.host_percent > self.settings.host_pressure_pct
                        or sample.gpu_percent > self.settings.gpu_pressure_pct
                    )
                    if pressure_requested:
                        reasons = []
                        if sample.host_percent > self.settings.host_pressure_pct:
                            reasons.append(f"host={sample.host_percent:.2f}%>{self.settings.host_pressure_pct}")
                        if sample.gpu_percent > self.settings.gpu_pressure_pct:
                            reasons.append(f"gpu={sample.gpu_percent:.2f}%>{self.settings.gpu_pressure_pct}")
                        print(f"\n[PRESSURE] Pause requested. {' '.join(reasons)}", flush=True)
                        try:
                            _pause_child(self.process, True)
                        except Exception as exc:
                            print(f"Failed to terminate for pause: {exc}")
                            # Source breaks the live poll and samples again
                            # immediately when the process remains live.
                            break
                        self.paused = True
                        # The source samples global Python usage after it has
                        # stopped the child process tree.
                        sample = _with_python_usage(sample, self.gpu_device_index)
                        self.host_peak_external = sample.external_host_mib
                        self.gpu_peak_external = sample.external_gpu_mib
                        break
                    self.sleep(max(0.1, self.settings.poll_seconds))

                return_code = self.process.poll()
                # A child that exits while stopped does not end the wrapper:
                # the reference continues waiting for gate reopening, then its
                # outer loop launches a replacement command.
                if self.paused:
                    continue
                if return_code is None:
                    continue
                finished_process = self.process
                self.process = None
                return int(getattr(finished_process, "returncode", return_code))
        except KeyboardInterrupt:
            if self.paused and self.process is not None:
                try:
                    _pause_child(self.process, False)
                except (OSError, ProcessLookupError):
                    pass
            if self.process is not None:
                terminate_standalone_pressure_tree(self.process)
            return 130
        finally:
            # The standalone source installs process-lifetime handlers and
            # clears its module-level child handle on exit. Keep that lifecycle
            # when the wrapper is called directly as well as through its CLI.
            self.process = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pause/continue one child under memory pressure")
    parser.add_argument("--host-pressure-pct", type=float, default=90)
    parser.add_argument("--host-resume-pct", type=float, default=85)
    parser.add_argument("--gpu-pressure-pct", type=float, default=90)
    parser.add_argument("--gpu-resume-pct", type=float, default=85)
    parser.add_argument("--external-memory-drop-mib", type=float, default=500)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--swap-pressure-pct", type=float, default=70)
    parser.add_argument("--swap-resume-pct", type=float, default=70)
    parser.add_argument("--gpu-device-index", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("No command provided")
    if not 0.0 <= args.host_resume_pct <= args.host_pressure_pct <= 100.0:
        raise SystemExit(
            "host RAM resume/pressure percentages must satisfy "
            "0 <= resume <= pressure <= 100"
        )
    if not 0.0 <= args.gpu_resume_pct <= args.gpu_pressure_pct <= 100.0:
        raise SystemExit(
            "GPU resume/pressure percentages must satisfy "
            "0 <= resume <= pressure <= 100"
        )
    if args.external_memory_drop_mib <= 0.0:
        raise SystemExit("--external-memory-drop-mib must be > 0")
    if args.poll_seconds <= 0.0:
        raise SystemExit("--poll-seconds must be > 0")
    settings = StandalonePressureSettings(
        host_pressure_pct=args.host_pressure_pct,
        host_resume_pct=args.host_resume_pct,
        gpu_pressure_pct=args.gpu_pressure_pct,
        gpu_resume_pct=args.gpu_resume_pct,
        external_memory_drop_mib=args.external_memory_drop_mib,
        poll_seconds=args.poll_seconds,
        swap_pressure_pct=args.swap_pressure_pct,
        swap_resume_pct=args.swap_resume_pct,
    )
    return PressureAwareWrapper(command, settings, gpu_device_index=args.gpu_device_index).run()


if __name__ == "__main__":
    raise SystemExit(main())
