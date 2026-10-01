"""Dependency-free compatibility with CORA continual-RL metrics/protocols.

The formulas mirror AGI-Labs/continual_rl develop branch semantics while
operating directly on numeric traces instead of TensorBoard event files.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CORAProtocol:
    """Canonical sequential experiment metadata from CORA."""

    name: str
    tasks: tuple[str, ...]
    steps_per_task: int
    cycles: int

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("CORA protocol requires at least one task")
        if self.steps_per_task < 1 or self.cycles < 1:
            raise ValueError("CORA steps_per_task and cycles must be positive")

    @property
    def total_steps(self) -> int:
        return len(self.tasks) * self.steps_per_task * self.cycles

    def training_regions(self, task_index: int) -> tuple[tuple[int, int], ...]:
        if not 0 <= task_index < len(self.tasks):
            raise IndexError("CORA task index out of range")
        regions: list[tuple[int, int]] = []
        tasks_per_cycle = len(self.tasks)
        for cycle in range(self.cycles):
            stage = cycle * tasks_per_cycle + task_index
            start = stage * self.steps_per_task
            regions.append((start, start + self.steps_per_task))
        return tuple(regions)


CORA_ATARI_6_TASKS_5_CYCLES = CORAProtocol(
    name="atari_6_tasks_5_cycles",
    tasks=(
        "SpaceInvadersNoFrameskip-v4",
        "KrullNoFrameskip-v4",
        "BeamRiderNoFrameskip-v4",
        "HeroNoFrameskip-v4",
        "StarGunnerNoFrameskip-v4",
        "MsPacmanNoFrameskip-v4",
    ),
    steps_per_task=50_000_000,
    cycles=5,
)

CORA_PROCGEN_6_TASKS_5_CYCLES = CORAProtocol(
    name="procgen_6_tasks_5_cycles",
    tasks=(
        "climber-v0",
        "dodgeball-v0",
        "ninja-v0",
        "starpilot-v0",
        "bigfish-v0",
        "fruitbot-v0",
    ),
    steps_per_task=5_000_000,
    cycles=5,
)


@dataclass(frozen=True)
class CORATrace:
    """One evaluation trace for one task in one run."""

    steps: np.ndarray
    returns: np.ndarray

    @classmethod
    def from_sequences(
        cls,
        steps: Sequence[float],
        returns: Sequence[float],
    ) -> "CORATrace":
        x = np.asarray(steps, dtype=np.float64)
        y = np.asarray(returns, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0:
            raise ValueError("CORA trace must contain matching non-empty vectors")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("CORA trace values must be finite")
        if np.any(np.diff(x) < 0):
            raise ValueError("CORA trace steps must be nondecreasing")
        return cls(steps=x, returns=y)


def _region(
    trace: CORATrace,
    low: float | None,
    high: float | None,
) -> np.ndarray:
    """Match CORA's strict x>low and x<high region selection."""

    lower = (
        trace.steps > low
        if low is not None
        else np.ones(trace.steps.shape, dtype=bool)
    )
    upper = (
        trace.steps < high
        if high is not None
        else np.ones(trace.steps.shape, dtype=bool)
    )
    values = trace.returns[lower & upper]
    if values.size == 0:
        raise ValueError(
            f"CORA metric region ({low}, {high}) contains no evaluations"
        )
    return values


def cora_return_scale(traces: Sequence[CORATrace]) -> float:
    """Return CORA's 1/max(abs(return)) task-specific scale."""

    if not traces:
        raise ValueError("at least one CORA trace is required")
    maximum = max(float(np.abs(trace.returns).max()) for trace in traces)
    if maximum == 0.0:
        raise ZeroDivisionError(
            "CORA normalization is undefined when all task returns are zero"
        )
    return 1.0 / maximum


def cora_isolated_forgetting(
    traces: Sequence[CORATrace],
    *,
    task_steps: int,
    task_id: int,
    num_tasks: int,
    num_cycles: int = 1,
    return_scale: float | None = None,
) -> dict[int, dict[int, list[float]]]:
    """Reproduce CORA's isolated-forgetting computation."""

    if min(task_steps, num_tasks, num_cycles) < 1:
        raise ValueError(
            "CORA task_steps, num_tasks and num_cycles must be positive"
        )
    if not 0 <= task_id < num_tasks:
        raise ValueError("CORA task_id is out of range")
    scale = cora_return_scale(traces) if return_scale is None else float(return_scale)
    output: dict[int, dict[int, list[float]]] = {
        index: {} for index in range(num_tasks)
    }

    for trace in traces:
        scaled = CORATrace(
            steps=trace.steps,
            returns=trace.returns * scale,
        )
        for cycle_id in range(num_cycles):
            for subsequent_task_id in range(num_tasks):
                if cycle_id == 0 and subsequent_task_id <= task_id:
                    continue
                offset = cycle_id * num_tasks
                stage = subsequent_task_id + offset
                pre_stage = _region(
                    scaled,
                    None,
                    stage * task_steps,
                )
                stage_values = _region(
                    scaled,
                    stage * task_steps,
                    (stage + 1) * task_steps,
                )
                forgetting = float(pre_stage[-1] - stage_values[-1])
                output[subsequent_task_id].setdefault(cycle_id, []).append(
                    forgetting
                )
    return output


def cora_isolated_zero_shot_forward_transfer(
    traces: Sequence[CORATrace],
    *,
    task_steps: int,
    prior_task_ids: Sequence[int],
    return_scale: float | None = None,
) -> dict[int, list[float]]:
    """Reproduce CORA's isolated zero-shot forward-transfer computation."""

    if task_steps < 1:
        raise ValueError("CORA task_steps must be positive")
    ids = tuple(int(index) for index in prior_task_ids)
    if any(index < 0 for index in ids):
        raise ValueError("CORA prior task ids must be non-negative")
    scale = cora_return_scale(traces) if return_scale is None else float(return_scale)
    output = {index: [] for index in ids}

    for trace in traces:
        scaled = CORATrace(
            steps=trace.steps,
            returns=trace.returns * scale,
        )
        initial = float(scaled.returns[0])
        for prior_task_id in ids:
            stage_values = _region(
                scaled,
                prior_task_id * task_steps,
                (prior_task_id + 1) * task_steps,
            )
            baseline = initial
            if prior_task_id > 0:
                baseline = float(
                    _region(
                        scaled,
                        None,
                        prior_task_id * task_steps,
                    )[-1]
                )
            output[prior_task_id].append(
                float(stage_values[-1] - baseline)
            )
    return output
