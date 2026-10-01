"""Continual World task evaluation and performance-matrix recording."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from rl_bgd.envs.continual_world.stream import ContinuousTaskEnv
from rl_bgd.metrics.continual import (
    backward_transfer,
    final_average_performance,
    forgetting,
)


class EvaluationAgent(Protocol):
    def act(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor: ...


@dataclass(frozen=True)
class TaskEvaluation:
    mean_return: float
    success_rate: float | None


@torch.no_grad()
def evaluate_task(
    agent: EvaluationAgent,
    env: ContinuousTaskEnv,
    *,
    episodes: int = 5,
    seed: int = 50_000,
    max_episode_steps: int = 200,
) -> TaskEvaluation:
    """Evaluate one task without exposing its identity to the agent."""

    if episodes < 1 or max_episode_steps < 1:
        raise ValueError("evaluation budget must be positive")
    returns: list[float] = []
    successes: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        episode_success = 0.0
        saw_success_signal = False
        for _ in range(max_episode_steps):
            action = agent.act(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if "success" in info:
                saw_success_signal = True
                episode_success = max(
                    episode_success,
                    float(info["success"]),
                )
            if terminated or truncated:
                break
        returns.append(episode_return)
        if saw_success_signal:
            successes.append(episode_success)

    return TaskEvaluation(
        mean_return=float(np.mean(returns)),
        success_rate=(float(np.mean(successes)) if successes else None),
    )


@torch.no_grad()
def evaluate_tasks(
    agent: EvaluationAgent,
    envs: Sequence[ContinuousTaskEnv],
    *,
    episodes: int = 5,
    seed: int = 50_000,
    max_episode_steps: int = 200,
) -> tuple[TaskEvaluation, ...]:
    """Evaluate a policy on every benchmark occurrence using separate envs."""

    return tuple(
        evaluate_task(
            agent,
            env,
            episodes=episodes,
            seed=seed + index * 10_000,
            max_episode_steps=max_episode_steps,
        )
        for index, env in enumerate(envs)
    )


class PerformanceMatrixRecorder:
    """Accumulate stage-by-task scores and compute standard CL summaries."""

    def __init__(
        self,
        task_names: Sequence[str],
    ) -> None:
        if not task_names:
            raise ValueError("at least one task name is required")
        self.task_names = tuple(task_names)
        self.stage_labels: list[str] = []
        self._rows: list[list[float]] = []

    def append(
        self,
        stage_label: str,
        scores: Sequence[float],
    ) -> None:
        row = np.asarray(scores, dtype=np.float64)
        if row.shape != (len(self.task_names),):
            raise ValueError("performance row width must match task count")
        if not np.isfinite(row).all():
            raise ValueError("performance row must be finite")
        self.stage_labels.append(stage_label)
        self._rows.append(row.tolist())

    @property
    def matrix(self) -> np.ndarray:
        if not self._rows:
            return np.empty(
                (0, len(self.task_names)),
                dtype=np.float64,
            )
        return np.asarray(self._rows, dtype=np.float64)

    def summary(self) -> dict[str, float | list[float]]:
        matrix = self.matrix
        tasks = len(self.task_names)
        if matrix.shape[0] < tasks:
            raise RuntimeError("a complete stage-by-task matrix is required for CL summary")
        square = matrix[:tasks, :]
        forgetting_values, mean_forgetting = forgetting(square.tolist())
        return {
            "final_average": final_average_performance(square[-1].tolist()),
            "mean_forgetting": mean_forgetting,
            "backward_transfer": backward_transfer(square.tolist()),
            "forgetting_by_task": forgetting_values.tolist(),
        }
