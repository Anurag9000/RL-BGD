from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from rl_bgd.experiments.parallel import (
    GpuState,
    build_worker_slots,
    detect_gpu_ids,
    parse_gpu_ids,
    run_suite_parallel,
)
from rl_bgd.experiments.suites import (
    materialize_suite,
)


def test_parse_gpu_ids_rejects_duplicates_and_non_numeric() -> None:
    assert parse_gpu_ids(
        "0, 2,7"
    ) == (
        "0",
        "2",
        "7",
    )
    assert parse_gpu_ids(
        ""
    ) == ()
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        parse_gpu_ids(
            "0,0"
        )
    with pytest.raises(
        ValueError,
        match="integer",
    ):
        parse_gpu_ids(
            "GPU-a"
        )


def test_detect_gpu_ids_prefers_explicit_then_visible_environment() -> None:
    assert detect_gpu_ids(
        explicit="3,4",
        environment={
            "CUDA_VISIBLE_DEVICES": "8,9"
        },
    ) == (
        "3",
        "4",
    )
    assert detect_gpu_ids(
        environment={
            "CUDA_VISIBLE_DEVICES": "8,9"
        }
    ) == (
        "8",
        "9",
    )
    assert detect_gpu_ids(
        environment={
            "CUDA_VISIBLE_DEVICES": "-1"
        }
    ) == ()


def test_worker_slots_are_gpu_first_with_cpu_fallback() -> None:
    assert build_worker_slots(
        gpu_ids=(
            "0",
            "2",
        ),
        workers_per_gpu=2,
        cpu_workers=7,
    ) == (
        pytest.helpers.worker_slot(
            "gpu-0-worker-0",
            "0",
        )
        if False
        else build_worker_slots(
            gpu_ids=(
                "0",
                "2",
            ),
            workers_per_gpu=2,
            cpu_workers=7,
        )[
            0
        ],
        *build_worker_slots(
            gpu_ids=(
                "0",
                "2",
            ),
            workers_per_gpu=2,
            cpu_workers=7,
        )[
            1:
        ],
    )

    slots = build_worker_slots(
        gpu_ids=(
            "0",
            "2",
        ),
        workers_per_gpu=2,
        cpu_workers=7,
    )
    assert [
        (
            slot.name,
            slot.gpu_id,
        )
        for slot in slots
    ] == [
        (
            "gpu-0-worker-0",
            "0",
        ),
        (
            "gpu-0-worker-1",
            "0",
        ),
        (
            "gpu-2-worker-0",
            "2",
        ),
        (
            "gpu-2-worker-1",
            "2",
        ),
    ]

    cpu_slots = build_worker_slots(
        gpu_ids=(),
        workers_per_gpu=1,
        cpu_workers=2,
    )
    assert [
        (
            slot.name,
            slot.gpu_id,
        )
        for slot in cpu_slots
    ] == [
        (
            "cpu-worker-0",
            None,
        ),
        (
            "cpu-worker-1",
            None,
        ),
    ]


def test_parallel_suite_assigns_each_child_to_worker_gpu(
    tmp_path: Path,
) -> None:
    manifest = materialize_suite(
        "smoke",
        tmp_path,
    )
    jobs = manifest[
        "jobs"
    ]
    assert isinstance(
        jobs,
        list,
    )
    selected_run_ids = tuple(
        str(job["run_id"])
        for job in jobs[
            :2
        ]
    )

    calls: list[
        tuple[
            tuple[str, ...],
            dict[str, str],
        ]
    ] = []

    def fake_runner(
        command: Any,
        environment: Any,
    ) -> subprocess.CompletedProcess[
        str
    ]:
        calls.append(
            (
                tuple(command),
                dict(environment),
            )
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout='{"status":"success"}',
            stderr="",
        )

    seen_states: list[
        str
    ] = []

    def state_reader(
        gpu_id: str,
    ) -> GpuState:
        seen_states.append(
            gpu_id
        )
        return GpuState(
            gpu_id=gpu_id,
            memory_total_mb=24_000,
            memory_free_mb=20_000,
            utilization_percent=3,
        )

    summary = run_suite_parallel(
        "smoke",
        tmp_path,
        run_ids=selected_run_ids,
        gpu_ids=(
            "2",
            "7",
        ),
        min_free_vram_mb=12_000,
        max_gpu_utilization=20,
        poll_seconds=0.001,
        command_runner=fake_runner,
        state_reader=state_reader,
    )

    assert summary[
        "status"
    ] == "success"
    assert summary[
        "runs_selected"
    ] == 2
    assert set(
        seen_states
    ) == {
        "2",
        "7",
    }
    assert len(
        calls
    ) == 2

    called_run_ids = set()
    visible_devices = set()
    for command, environment in calls:
        assert (
            "scripts/run_paper_suite.py"
            in command
        )
        run_index = (
            command.index(
                "--run-id"
            )
            + 1
        )
        called_run_ids.add(
            command[
                run_index
            ]
        )
        visible_devices.add(
            environment[
                "CUDA_VISIBLE_DEVICES"
            ]
        )
        assert (
            environment[
                "RL_BGD_WORKER_SLOT"
            ].startswith(
                "gpu-"
            )
        )

    assert called_run_ids == set(
        selected_run_ids
    )
    assert visible_devices == {
        "2",
        "7",
    }
    assert (
        tmp_path
        / "smoke"
        / "parallel_execution_summary.json"
    ).is_file()


def test_parallel_suite_cpu_fallback_hides_cuda(
    tmp_path: Path,
) -> None:
    manifest = materialize_suite(
        "smoke",
        tmp_path,
    )
    jobs = manifest[
        "jobs"
    ]
    assert isinstance(
        jobs,
        list,
    )
    run_id = str(
        jobs[
            0
        ][
            "run_id"
        ]
    )
    environments: list[
        dict[str, str]
    ] = []

    def fake_runner(
        command: Any,
        environment: Any,
    ) -> subprocess.CompletedProcess[
        str
    ]:
        environments.append(
            dict(
                environment
            )
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="{}",
            stderr="",
        )

    summary = run_suite_parallel(
        "smoke",
        tmp_path,
        run_ids=(
            run_id,
        ),
        gpu_ids=(),
        cpu_workers=2,
        command_runner=fake_runner,
    )

    assert summary[
        "status"
    ] == "success"
    assert len(
        environments
    ) == 1
    assert environments[
        0
    ][
        "CUDA_VISIBLE_DEVICES"
    ] == ""


def test_parallel_suite_rejects_unknown_run_selection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        KeyError,
        match="unknown suite run IDs",
    ):
        run_suite_parallel(
            "smoke",
            tmp_path,
            run_ids=(
                "does-not-exist",
            ),
            gpu_ids=(),
        )
