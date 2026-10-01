"""Task-aware Continual World protocol matching the original benchmark interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_bgd.envs.continual_world.stream import ContinuousTaskEnv
from rl_bgd.envs.synthetic.lqr import TensorBox


class TaskIdentityObservationEnv:
    """Append the benchmark occurrence ID as a one-hot observation suffix."""

    def __init__(
        self,
        env: ContinuousTaskEnv,
        *,
        task_index: int,
        num_tasks: int,
    ) -> None:
        if num_tasks < 1 or not 0 <= task_index < num_tasks:
            raise ValueError("invalid task identity dimensions")
        self.env = env
        self.task_index = task_index
        self.num_tasks = num_tasks
        self.action_space = env.action_space
        self.observation_space = TensorBox(
            low=torch.cat(
                [
                    env.observation_space.low,
                    torch.zeros(
                        num_tasks,
                        device=env.observation_space.low.device,
                        dtype=env.observation_space.low.dtype,
                    ),
                ]
            ),
            high=torch.cat(
                [
                    env.observation_space.high,
                    torch.ones(
                        num_tasks,
                        device=env.observation_space.high.device,
                        dtype=env.observation_space.high.dtype,
                    ),
                ]
            ),
        )

    def _augment(self, observation: Tensor) -> Tensor:
        identity = torch.zeros(
            self.num_tasks,
            device=observation.device,
            dtype=observation.dtype,
        )
        identity[self.task_index] = 1.0
        return torch.cat([observation, identity])

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        observation, info = self.env.reset(seed=seed)
        return self._augment(observation), dict(info)

    def step(
        self,
        action: Tensor,
    ) -> tuple[Tensor, float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._augment(observation),
            reward,
            terminated,
            truncated,
            dict(info),
        )


@dataclass(frozen=True)
class CanonicalContinualWorldConfig:
    """Published Continual World stage budget controls."""

    steps_per_task: int = 1_000_000

    def validate(self) -> None:
        if self.steps_per_task < 1:
            raise ValueError("steps_per_task must be positive")


class CanonicalContinualWorldStreamEnv:
    """Task-aware CW stream with occurrence ID and explicit stage boundaries.

    This path intentionally exposes task identity and stage changes. It is an
    oracle/canonical comparison and must never be used as the strict
    task-agnostic protocol.
    """

    def __init__(
        self,
        envs: list[ContinuousTaskEnv],
        task_names: list[str],
        *,
        config: CanonicalContinualWorldConfig | None = None,
    ) -> None:
        if not envs:
            raise ValueError("Continual World stream requires at least one task")
        if len(envs) != len(task_names):
            raise ValueError("task environment/name counts must match")
        self.config = config or CanonicalContinualWorldConfig()
        self.config.validate()
        self.envs = list(envs)
        self.task_names = list(task_names)
        self.num_envs = len(envs)
        self.steps_per_env = self.config.steps_per_task
        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space
        for env in envs[1:]:
            if env.action_space.shape != self.action_space.shape:
                raise ValueError("Continual World action dimensions differ")
            if env.observation_space.shape != self.observation_space.shape:
                raise ValueError("Continual World observation dimensions differ")

        self.cur_seq_idx = 0
        self.task_step = 0
        self.environment_step = 0

    @property
    def total_step_limit(self) -> int:
        return self.num_envs * self.config.steps_per_task

    @property
    def evaluation_context(self) -> dict[str, int | str]:
        return {
            "task_index": self.cur_seq_idx,
            "task_name": self.task_names[self.cur_seq_idx],
            "task_step": self.task_step,
            "environment_step": self.environment_step,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        observation, _ = self.envs[self.cur_seq_idx].reset(seed=seed)
        return observation, {
            "seq_idx": self.cur_seq_idx,
            "task_name": self.task_names[self.cur_seq_idx],
        }

    def step(
        self,
        action: Tensor,
    ) -> tuple[Tensor, float, bool, bool, dict[str, object]]:
        if self.environment_step >= self.total_step_limit:
            raise RuntimeError("Continual World stream is exhausted")

        current_index = self.cur_seq_idx
        observation, reward, terminated, truncated, raw_info = self.envs[
            current_index
        ].step(action)
        self.environment_step += 1
        self.task_step += 1

        info: dict[str, object] = {
            "seq_idx": current_index,
            "task_name": self.task_names[current_index],
        }
        if "success" in raw_info:
            info["success"] = raw_info["success"]

        if self.task_step >= self.config.steps_per_task:
            # The published benchmark forces an episode boundary at each stage.
            truncated = True
            info["TimeLimit.truncated"] = True
            if current_index + 1 < self.num_envs:
                self.cur_seq_idx += 1
                self.task_step = 0

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
