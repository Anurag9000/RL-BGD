"""GPU-first parallel orchestration for canonical paper-suite runs."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rl_bgd.experiments.suites import materialize_suite


@dataclass(frozen=True)
class GpuState:
    gpu_id: str
    memory_total_mb: int
    memory_free_mb: int
    utilization_percent: int


@dataclass(frozen=True)
class WorkerSlot:
    name: str
    gpu_id: str | None


@dataclass(frozen=True)
class ParallelRunResult:
    run_id: str
    worker: str
    gpu_id: str | None
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str


CommandRunner = Callable[
    [
        Sequence[str],
        Mapping[str, str],
    ],
    subprocess.CompletedProcess[str],
]


def _default_command_runner(
    command: Sequence[str],
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )


def parse_gpu_ids(
    value: str | None,
) -> tuple[str, ...]:
    """Parse an explicit comma-separated CUDA device list."""

    if value is None:
        return ()
    stripped = value.strip()
    if not stripped:
        return ()
    ids = tuple(
        part.strip()
        for part in stripped.split(",")
        if part.strip()
    )
    if len(ids) != len(
        set(ids)
    ):
        raise ValueError(
            "GPU IDs must be unique"
        )
    if any(
        not gpu_id.isdigit()
        for gpu_id in ids
    ):
        raise ValueError(
            "GPU IDs must be non-negative integer strings"
        )
    return ids


def detect_gpu_ids(
    *,
    explicit: str | None = None,
    environment: Mapping[
        str,
        str,
    ] | None = None,
) -> tuple[str, ...]:
    """Resolve physical GPU IDs without importing CUDA libraries."""

    parsed = parse_gpu_ids(
        explicit
    )
    if parsed:
        return parsed

    env = (
        environment
        if environment is not None
        else os.environ
    )
    visible = env.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if visible is not None:
        if visible.strip() in {
            "",
            "-1",
        }:
            return ()
        return parse_gpu_ids(
            visible
        )

    if shutil.which(
        "nvidia-smi"
    ) is None:
        return ()

    completed = subprocess.run(
        [
            "nvidia-smi",
            (
                "--query-gpu=index"
            ),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return ()
    return tuple(
        line.strip()
        for line in (
            completed.stdout.splitlines()
        )
        if line.strip().isdigit()
    )


def query_gpu_state(
    gpu_id: str,
) -> GpuState:
    """Read the current free memory/utilization for one physical GPU."""

    if shutil.which(
        "nvidia-smi"
    ) is None:
        raise RuntimeError(
            "nvidia-smi is unavailable"
        )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            (
                "--query-gpu="
                "memory.total,memory.free,"
                "utilization.gpu"
            ),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi failed for GPU "
            f"{gpu_id}: "
            f"{completed.stderr.strip()}"
        )
    line = completed.stdout.strip().splitlines()
    if len(line) != 1:
        raise RuntimeError(
            "nvidia-smi returned an unexpected GPU row count"
        )
    fields = [
        field.strip()
        for field in line[
            0
        ].split(",")
    ]
    if len(fields) != 3:
        raise RuntimeError(
            "nvidia-smi returned an unexpected GPU state format"
        )
    try:
        total, free, utilization = (
            int(field)
            for field in fields
        )
    except ValueError as exc:
        raise RuntimeError(
            "nvidia-smi returned non-integer GPU state"
        ) from exc
    return GpuState(
        gpu_id=gpu_id,
        memory_total_mb=total,
        memory_free_mb=free,
        utilization_percent=(
            utilization
        ),
    )


def build_worker_slots(
    *,
    gpu_ids: Sequence[str],
    workers_per_gpu: int,
    cpu_workers: int,
) -> tuple[WorkerSlot, ...]:
    """Build deterministic worker slots with GPU-first CPU fallback."""

    if workers_per_gpu < 1:
        raise ValueError(
            "workers_per_gpu must be positive"
        )
    if cpu_workers < 1:
        raise ValueError(
            "cpu_workers must be positive"
        )

    if gpu_ids:
        return tuple(
            WorkerSlot(
                name=(
                    f"gpu-{gpu_id}-"
                    f"worker-{worker_index}"
                ),
                gpu_id=gpu_id,
            )
            for gpu_id in gpu_ids
            for worker_index in range(
                workers_per_gpu
            )
        )
    return tuple(
        WorkerSlot(
            name=(
                f"cpu-worker-{index}"
            ),
            gpu_id=None,
        )
        for index in range(
            cpu_workers
        )
    )


def _selected_runs(
    manifest: Mapping[str, object],
    *,
    job_ids: Sequence[str],
    seeds: Sequence[int],
    run_ids: Sequence[str],
) -> tuple[
    dict[str, Any],
    ...,
]:
    raw_jobs = manifest.get(
        "jobs"
    )
    if not isinstance(
        raw_jobs,
        list,
    ):
        raise TypeError(
            "suite manifest jobs must be a list"
        )
    jobs = [
        job
        for job in raw_jobs
        if isinstance(
            job,
            dict,
        )
    ]
    if len(jobs) != len(
        raw_jobs
    ):
        raise TypeError(
            "suite manifest contains a non-mapping job"
        )

    requested_job_ids = set(
        job_ids
    )
    requested_seeds = set(
        seeds
    )
    requested_run_ids = set(
        run_ids
    )

    known_job_ids = {
        str(job["job_id"])
        for job in jobs
    }
    known_seeds = {
        int(job["seed"])
        for job in jobs
    }
    known_run_ids = {
        str(job["run_id"])
        for job in jobs
    }

    unknown_job_ids = (
        requested_job_ids
        - known_job_ids
    )
    unknown_seeds = (
        requested_seeds
        - known_seeds
    )
    unknown_run_ids = (
        requested_run_ids
        - known_run_ids
    )
    if unknown_job_ids:
        raise KeyError(
            "unknown suite job IDs: "
            + ", ".join(
                sorted(
                    unknown_job_ids
                )
            )
        )
    if unknown_seeds:
        raise KeyError(
            "unknown suite seeds: "
            + ", ".join(
                str(value)
                for value in sorted(
                    unknown_seeds
                )
            )
        )
    if unknown_run_ids:
        raise KeyError(
            "unknown suite run IDs: "
            + ", ".join(
                sorted(
                    unknown_run_ids
                )
            )
        )

    selected = tuple(
        job
        for job in jobs
        if (
            not requested_job_ids
            or str(
                job["job_id"]
            )
            in requested_job_ids
        )
        and (
            not requested_seeds
            or int(
                job["seed"]
            )
            in requested_seeds
        )
        and (
            not requested_run_ids
            or str(
                job["run_id"]
            )
            in requested_run_ids
        )
    )
    if (
        (
            requested_job_ids
            or requested_seeds
            or requested_run_ids
        )
        and not selected
    ):
        raise ValueError(
            "suite filters select no runs"
        )
    return selected


def _wait_for_gpu(
    gpu_id: str,
    *,
    min_free_vram_mb: int,
    max_gpu_utilization: int,
    poll_seconds: float,
    state_reader: Callable[
        [str],
        GpuState,
    ],
) -> None:
    if min_free_vram_mb < 0:
        raise ValueError(
            "min_free_vram_mb must be non-negative"
        )
    if not (
        0
        <= max_gpu_utilization
        <= 100
    ):
        raise ValueError(
            "max_gpu_utilization must lie in [0, 100]"
        )
    if poll_seconds <= 0:
        raise ValueError(
            "poll_seconds must be positive"
        )
    if (
        min_free_vram_mb == 0
        and max_gpu_utilization
        == 100
    ):
        return

    while True:
        state = state_reader(
            gpu_id
        )
        if (
            state.memory_free_mb
            >= min_free_vram_mb
            and state.utilization_percent
            <= max_gpu_utilization
        ):
            return
        time.sleep(
            poll_seconds
        )


def _child_command(
    *,
    suite_name: str,
    output_root: str | Path,
    run_id: str,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_paper_suite.py",
        suite_name,
        "--output-root",
        str(
            output_root
        ),
        "--execute",
        "--run-id",
        run_id,
    ]
    if not resume:
        command.append(
            "--no-resume"
        )
    return command


def run_suite_parallel(
    suite_name: str,
    output_root: str | Path,
    *,
    job_ids: Sequence[str] = (),
    seeds: Sequence[int] = (),
    run_ids: Sequence[str] = (),
    gpu_ids: Sequence[str] = (),
    workers_per_gpu: int = 1,
    cpu_workers: int = 1,
    min_free_vram_mb: int = 0,
    max_gpu_utilization: int = 100,
    poll_seconds: float = 30.0,
    resume: bool = True,
    command_runner: CommandRunner | None = None,
    state_reader: Callable[
        [str],
        GpuState,
    ] = query_gpu_state,
) -> dict[str, object]:
    """Execute selected canonical suite runs concurrently across worker slots."""

    manifest = materialize_suite(
        suite_name,
        output_root,
    )
    selected = _selected_runs(
        manifest,
        job_ids=job_ids,
        seeds=seeds,
        run_ids=run_ids,
    )
    slots = build_worker_slots(
        gpu_ids=gpu_ids,
        workers_per_gpu=(
            workers_per_gpu
        ),
        cpu_workers=cpu_workers,
    )
    if not slots:
        raise RuntimeError(
            "parallel suite has no worker slots"
        )

    run_queue: queue.Queue[
        dict[str, Any]
    ] = queue.Queue()
    for job in selected:
        run_queue.put(
            job
        )

    runner = (
        command_runner
        if command_runner is not None
        else _default_command_runner
    )
    results: list[
        ParallelRunResult
    ] = []
    results_lock = (
        threading.Lock()
    )

    def worker(
        slot: WorkerSlot,
    ) -> None:
        while True:
            try:
                job = run_queue.get_nowait()
            except queue.Empty:
                return

            run_id = str(
                job["run_id"]
            )
            started = (
                time.perf_counter()
            )
            try:
                if slot.gpu_id is not None:
                    _wait_for_gpu(
                        slot.gpu_id,
                        min_free_vram_mb=(
                            min_free_vram_mb
                        ),
                        max_gpu_utilization=(
                            max_gpu_utilization
                        ),
                        poll_seconds=(
                            poll_seconds
                        ),
                        state_reader=(
                            state_reader
                        ),
                    )

                environment = dict(
                    os.environ
                )
                if slot.gpu_id is None:
                    environment[
                        "CUDA_VISIBLE_DEVICES"
                    ] = ""
                else:
                    environment[
                        "CUDA_VISIBLE_DEVICES"
                    ] = slot.gpu_id
                environment[
                    "RL_BGD_WORKER_SLOT"
                ] = slot.name

                command = _child_command(
                    suite_name=(
                        suite_name
                    ),
                    output_root=(
                        output_root
                    ),
                    run_id=run_id,
                    resume=resume,
                )
                completed = runner(
                    command,
                    environment,
                )
                result = (
                    ParallelRunResult(
                        run_id=run_id,
                        worker=slot.name,
                        gpu_id=(
                            slot.gpu_id
                        ),
                        returncode=(
                            completed.returncode
                        ),
                        duration_seconds=(
                            time.perf_counter()
                            - started
                        ),
                        stdout=(
                            completed.stdout
                        ),
                        stderr=(
                            completed.stderr
                        ),
                    )
                )
            except Exception as exc:
                result = (
                    ParallelRunResult(
                        run_id=run_id,
                        worker=slot.name,
                        gpu_id=(
                            slot.gpu_id
                        ),
                        returncode=-1,
                        duration_seconds=(
                            time.perf_counter()
                            - started
                        ),
                        stdout="",
                        stderr=(
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    )
                )
            finally:
                with results_lock:
                    results.append(
                        result
                    )
                run_queue.task_done()

    threads = [
        threading.Thread(
            target=worker,
            args=(slot,),
            name=slot.name,
        )
        for slot in slots
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ordered = sorted(
        results,
        key=lambda result: (
            result.run_id
        ),
    )
    failures = [
        result.run_id
        for result in ordered
        if result.returncode != 0
    ]

    summary = {
        "schema_version": 1,
        "suite": suite_name,
        "manifest_path": (
            manifest[
                "manifest_path"
            ]
        ),
        "runs_selected": len(
            selected
        ),
        "worker_slots": [
            asdict(
                slot
            )
            for slot in slots
        ],
        "gpu_pressure_policy": {
            "min_free_vram_mb": (
                min_free_vram_mb
            ),
            "max_gpu_utilization": (
                max_gpu_utilization
            ),
            "poll_seconds": (
                poll_seconds
            ),
        },
        "resume": resume,
        "results": [
            asdict(
                result
            )
            for result in ordered
        ],
        "failures": failures,
        "status": (
            "success"
            if not failures
            else "failed"
        ),
    }

    suite_dir = (
        Path(output_root)
        / suite_name
    )
    suite_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path = (
        suite_dir
        / "parallel_execution_summary.json"
    )
    temporary = (
        summary_path.with_name(
            (
                f".{summary_path.name}."
                f"{os.getpid()}.tmp"
            )
        )
    )
    try:
        temporary.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(
            temporary,
            summary_path,
        )
    finally:
        temporary.unlink(
            missing_ok=True
        )
    summary[
        "summary_path"
    ] = str(
        summary_path
    )
    return summary
