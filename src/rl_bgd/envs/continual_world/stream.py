"""Strict task-agnostic Continual World sequence and stream semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from torch import Tensor

from rl_bgd.envs.synthetic.lqr import TensorBox

ContinualWorldBenchmark = Literal["CW10", "CW20"]

# Canonical order from the published Continual World benchmark.
CW10_TASKS_V1: tuple[str, ...] = (
    "hammer-v1",
    "push-wall-v1",
    "faucet-close-v1",
    "push-back-v1",
    "stick-pull-v1",
    "handle-press-side-v1",
    "push-v1",
    "shelf-place-v1",
    "window-close-v1",
    "peg-unplug-side-v1",
)


def _modernize_task_name(task_name: str) -> str:
    if not task_name.endswith("-v1"):
        raise ValueError("canonical Continual World task names must use the v1 suffix")
    return task_name.removesuffix("-v1") + "-v3"


CW10_TASKS_V3: tuple[str, ...] = tuple(_modernize_task_name(name) for name in CW10_TASKS_V1)
CW20_TASKS_V3: tuple[str, ...] = CW10_TASKS_V3 + CW10_TASKS_V3


def continual_world_task_sequence(
    benchmark: ContinualWorldBenchmark,
) -> tuple[str, ...]:
    """Return the canonical CW10/CW20 order using modern Meta-World IDs."""

    if benchmark == "CW10":
        return CW10_TASKS_V3
    if benchmark == "CW20":
        return CW20_TASKS_V3
    raise ValueError(f"unsupported Continual World benchmark: {benchmark}")


class ContinuousTaskEnv(Protocol):
    @property
    def action_space(self) -> TensorBox: ...

    @property
    def observation_space(self) -> TensorBox: ...

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, object]]: ...

    def step(
        self,
        action: Tensor,
    ) -> tuple[
        Tensor,
        float,
        bool,
        bool,
        dict[str, object],
    ]: ...


@dataclass(frozen=True)
class ContinualWorldStreamConfig:
    """Protocol controls for a hidden-boundary continual task sequence."""

    steps_per_task: int = 1_000_000
    strict_task_agnostic: bool = True

    def validate(self) -> None:
        if self.steps_per_task < 1:
            raise ValueError("steps_per_task must be positive")
        if not self.strict_task_agnostic:
            raise ValueError(
                "ContinualWorldStreamEnv only implements the strict task-agnostic protocol"
            )


class ContinualWorldStreamEnv:
    """Compose task environments without exposing task IDs or switch callbacks.

    When a task budget expires, the next task is reset internally and its first
    observation becomes the transition's next observation. If the previous task
    also ended naturally, that observation is cached so the caller's subsequent
    reset returns the same state instead of resetting twice. No synthetic
    termination/truncation flag is emitted solely because a task changed.
    """

    def __init__(
        self,
        envs: Sequence[ContinuousTaskEnv],
        task_names: Sequence[str],
        *,
        config: ContinualWorldStreamConfig | None = None,
    ) -> None:
        if not envs:
            raise ValueError("Continual World stream requires at least one task")
        if len(envs) != len(task_names):
            raise ValueError("task environment/name counts must match")
        self.config = config or ContinualWorldStreamConfig()
        self.config.validate()
        self.envs = list(envs)
        self.task_names = list(task_names)
        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space
        for env in envs[1:]:
            if env.action_space.shape != self.action_space.shape:
                raise ValueError("Continual World action dimensions differ")
            if env.observation_space.shape != self.observation_space.shape:
                raise ValueError("Continual World observation dimensions differ")

        self.task_index = 0
        self.task_step = 0
        self.environment_step = 0
        self._pending_reset_observation: Tensor | None = None
        self._internal_reset_counter = 0

    @property
    def total_step_limit(self) -> int:
        return len(self.envs) * self.config.steps_per_task

    @property
    def evaluation_context(
        self,
    ) -> dict[str, int | str]:
        """Evaluator-only context; never returned by reset/step."""

        return {
            "task_index": self.task_index,
            "task_name": self.task_names[self.task_index],
            "task_step": self.task_step,
            "environment_step": self.environment_step,
        }

    def _advance_task(self) -> None:
        if self.task_index + 1 >= len(self.envs):
            raise RuntimeError("Continual World stream is exhausted")
        self.task_index += 1
        self.task_step = 0

    def _internal_reset(
        self,
    ) -> Tensor:
        self._internal_reset_counter += 1
        observation, _ = self.envs[self.task_index].reset(
            seed=(1_000_000 + self._internal_reset_counter)
        )
        return observation

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        if self.environment_step >= (self.total_step_limit):
            # Trainers commonly reset immediately after a final natural
            # truncation before their outer step loop notices completion.
            # Permit that reset, but keep any further step invalid.
            observation, _ = self.envs[self.task_index].reset(seed=seed)
            return observation, {}
        if self._pending_reset_observation is not None:
            observation = self._pending_reset_observation
            self._pending_reset_observation = None
            return observation, {}
        observation, _ = self.envs[self.task_index].reset(seed=seed)
        # Strict path intentionally strips task/context metadata.
        return observation, {}

    def step(
        self,
        action: Tensor,
    ) -> tuple[
        Tensor,
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        if self.environment_step >= (self.total_step_limit):
            raise RuntimeError("Continual World stream is exhausted")
        (
            observation,
            reward,
            terminated,
            truncated,
            _,
        ) = self.envs[self.task_index].step(action)

        self.environment_step += 1
        self.task_step += 1
        hit_task_budget = self.task_step >= self.config.steps_per_task
        has_next_task = self.task_index + 1 < len(self.envs)

        if hit_task_budget and has_next_task:
            self._advance_task()
            observation = self._internal_reset()
            if terminated or truncated:
                self._pending_reset_observation = observation

        # Do not return task IDs, names, indices, context, or switch flags.
        return (
            observation,
            reward,
            terminated,
            truncated,
            {},
        )

    def close(self) -> None:
        """Close every task environment owned by the stream."""

        for env in self.envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()
