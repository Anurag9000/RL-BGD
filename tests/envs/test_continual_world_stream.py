import pytest
import torch

from rl_bgd.envs.continual_world import (
    CW10_TASKS_V1,
    CW10_TASKS_V3,
    CW20_TASKS_V3,
    ContinualWorldStreamConfig,
    ContinualWorldStreamEnv,
    continual_world_task_sequence,
)
from rl_bgd.envs.synthetic.lqr import TensorBox


class FakeTaskEnv:
    def __init__(
        self,
        value: float,
        *,
        horizon: int = 100,
    ) -> None:
        self.value = value
        self.horizon = horizon
        self.action_space = TensorBox(
            low=torch.tensor([-1.0]),
            high=torch.tensor([1.0]),
        )
        self.observation_space = TensorBox(
            low=torch.tensor([-10.0]),
            high=torch.tensor([10.0]),
        )
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, object],
    ]:
        del seed
        self._step = 0
        return (
            torch.tensor([self.value]),
            {
                "task_id": self.value,
                "env_name": f"task-{self.value}",
            },
        )

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        del action
        self._step += 1
        return (
            torch.tensor([self.value]),
            self.value,
            False,
            self._step >= self.horizon,
            {
                "task_id": self.value,
                "seq_idx": int(self.value),
            },
        )


def test_cw_sequences_preserve_official_order_and_repeat() -> None:
    assert CW10_TASKS_V1 == (
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
    assert tuple(
        name.removesuffix("-v1") + "-v3"
        for name in CW10_TASKS_V1
    ) == CW10_TASKS_V3
    assert (
        CW10_TASKS_V3 + CW10_TASKS_V3
    ) == CW20_TASKS_V3
    assert continual_world_task_sequence(
        "CW10"
    ) == CW10_TASKS_V3
    assert continual_world_task_sequence(
        "CW20"
    ) == CW20_TASKS_V3


def test_hidden_task_switch_emits_no_boundary_or_identity_signal() -> None:
    env = ContinualWorldStreamEnv(
        [
            FakeTaskEnv(1.0),
            FakeTaskEnv(2.0),
        ],
        [
            "first-v3",
            "second-v3",
        ],
        config=ContinualWorldStreamConfig(
            steps_per_task=2
        ),
    )
    observation, info = env.reset(seed=7)
    torch.testing.assert_close(
        observation,
        torch.tensor([1.0]),
    )
    assert info == {}

    _, _, terminated, truncated, info = env.step(
        torch.zeros(1)
    )
    assert not terminated
    assert not truncated
    assert info == {}

    observation, _, terminated, truncated, info = env.step(
        torch.zeros(1)
    )
    torch.testing.assert_close(
        observation,
        torch.tensor([2.0]),
    )
    assert not terminated
    assert not truncated
    assert info == {}
    assert env.evaluation_context == {
        "task_index": 1,
        "task_name": "second-v3",
        "task_step": 0,
        "environment_step": 2,
    }


def test_natural_episode_end_can_carry_pending_hidden_switch() -> None:
    env = ContinualWorldStreamEnv(
        [
            FakeTaskEnv(
                1.0,
                horizon=2,
            ),
            FakeTaskEnv(2.0),
        ],
        [
            "first-v3",
            "second-v3",
        ],
        config=ContinualWorldStreamConfig(
            steps_per_task=2
        ),
    )
    env.reset(seed=9)
    env.step(torch.zeros(1))
    _, _, terminated, truncated, info = env.step(
        torch.zeros(1)
    )
    assert not terminated
    assert truncated
    assert info == {}
    assert env.evaluation_context[
        "task_index"
    ] == 0

    observation, info = env.reset()
    torch.testing.assert_close(
        observation,
        torch.tensor([2.0]),
    )
    assert info == {}
    assert env.evaluation_context[
        "task_index"
    ] == 1


def test_stream_rejects_steps_after_total_budget() -> None:
    env = ContinualWorldStreamEnv(
        [
            FakeTaskEnv(1.0),
            FakeTaskEnv(2.0),
        ],
        [
            "first-v3",
            "second-v3",
        ],
        config=ContinualWorldStreamConfig(
            steps_per_task=1
        ),
    )
    env.reset(seed=11)
    env.step(torch.zeros(1))
    env.step(torch.zeros(1))

    # A trainer may issue its routine episode reset on the exact final step.
    observation, info = env.reset()
    torch.testing.assert_close(
        observation,
        torch.tensor([2.0]),
    )
    assert info == {}

    with pytest.raises(
        RuntimeError,
        match="exhausted",
    ):
        env.step(torch.zeros(1))
