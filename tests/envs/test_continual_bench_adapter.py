from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from rl_bgd.envs.continual_bench import (
    ContinualBenchStreamConfig,
    ContinualBenchStreamEnv,
)


class FakeBox:
    def __init__(
        self,
        low: list[float],
        high: list[float],
    ) -> None:
        self.low = np.asarray(
            low,
            dtype=np.float32,
        )
        self.high = np.asarray(
            high,
            dtype=np.float32,
        )
        self.shape = self.low.shape


class FakeContinualBench:
    all_tasks = [
        "button",
        "door",
        "window",
        "faucet",
        "peg",
        "block",
    ]

    def __init__(
        self,
        *,
        five_value: bool = False,
        success_on_step: int | None = None,
    ) -> None:
        self.action_space = FakeBox(
            [-1.0],
            [1.0],
        )
        self.observation_space = FakeBox(
            [-10.0, -10.0],
            [10.0, 10.0],
        )
        self.task = "button"
        self.steps = 0
        self.five_value = five_value
        self.success_on_step = success_on_step

    def set_task(
        self,
        task: str,
    ) -> None:
        self.task = task

    def reset(
        self,
        seed: int | None = None,
    ) -> np.ndarray:
        del seed
        return np.asarray(
            [
                float(self.all_tasks.index(self.task)),
                0.0,
            ],
            dtype=np.float32,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Any, ...]:
        assert action.shape == (1,)
        self.steps += 1
        success = (
            self.success_on_step
            is not None
            and self.steps
            == self.success_on_step
        )
        observation = np.asarray(
            [
                float(self.all_tasks.index(self.task)),
                float(self.steps),
            ],
            dtype=np.float32,
        )
        rewards = {
            name: float(index + 1)
            for index, name in enumerate(
                self.all_tasks
            )
        }
        info = {
            name: {
                "success": (
                    success
                    if name == self.task
                    else False
                )
            }
            for name in self.all_tasks
        }
        if self.five_value:
            return (
                observation,
                rewards,
                False,
                False,
                info,
            )
        return (
            observation,
            rewards,
            False,
            info,
        )


def test_fixed_step_switch_hides_task_and_selects_active_reward() -> None:
    base = FakeContinualBench()
    env = ContinualBenchStreamEnv(
        base,
        config=ContinualBenchStreamConfig(
            task_sequence=(
                "button",
                "door",
            ),
            switch_mode="fixed_steps",
            steps_per_task=2,
        ),
    )
    observation, info = env.reset(seed=0)
    assert observation.shape == (2,)
    assert info == {}

    _, reward, terminated, truncated, info = env.step(
        torch.tensor([0.0])
    )
    assert reward == pytest.approx(1.0)
    assert not terminated
    assert not truncated
    assert info == {}

    observation, reward, _, _, info = env.step(
        torch.tensor([0.0])
    )
    assert reward == pytest.approx(1.0)
    assert info == {}
    assert env.evaluation_context["task_name"] == "door"
    assert observation[0].item() == pytest.approx(1.0)


def test_success_switch_supports_five_value_api() -> None:
    env = ContinualBenchStreamEnv(
        FakeContinualBench(
            five_value=True,
            success_on_step=1,
        ),
        config=ContinualBenchStreamConfig(
            task_sequence=(
                "button",
                "door",
            ),
            switch_mode="success",
        ),
    )
    env.reset()
    observation, reward, terminated, truncated, info = env.step(
        torch.tensor([0.0])
    )
    assert reward == pytest.approx(1.0)
    assert not terminated
    assert not truncated
    assert info == {}
    assert env.evaluation_context["task_name"] == "door"
    assert observation[0].item() == pytest.approx(1.0)
