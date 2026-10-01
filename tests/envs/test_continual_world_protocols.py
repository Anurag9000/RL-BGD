import numpy as np
import pytest
import torch

from rl_bgd.envs.continual_world.canonical import (
    CanonicalContinualWorldConfig,
    CanonicalContinualWorldStreamEnv,
    TaskIdentityObservationEnv,
)
from rl_bgd.envs.continual_world.evaluation import (
    PerformanceMatrixRecorder,
    evaluate_task,
)
from rl_bgd.envs.synthetic.lqr import TensorBox


class FakeTaskEnv:
    def __init__(self, value: float) -> None:
        self.value = value
        self.action_space = TensorBox(
            low=torch.tensor([-1.0]),
            high=torch.tensor([1.0]),
        )
        self.observation_space = TensorBox(
            low=torch.tensor([-10.0]),
            high=torch.tensor([10.0]),
        )
        self.step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        del seed
        self.step_count = 0
        return torch.tensor([self.value]), {}

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, object]]:
        del action
        self.step_count += 1
        return (
            torch.tensor([self.value]),
            self.value,
            False,
            self.step_count >= 2,
            {"success": float(self.value > 0)},
        )


class ZeroAgent:
    def act(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        del observation, deterministic
        return torch.zeros(1)


def test_task_identity_wrapper_appends_occurrence_one_hot() -> None:
    env = TaskIdentityObservationEnv(
        FakeTaskEnv(2.0),
        task_index=1,
        num_tasks=3,
    )
    observation, _ = env.reset(seed=1)
    torch.testing.assert_close(
        observation,
        torch.tensor([2.0, 0.0, 1.0, 0.0]),
    )
    assert env.observation_space.shape == (4,)


def test_canonical_stream_exposes_identity_and_forces_stage_boundary() -> None:
    envs = [
        TaskIdentityObservationEnv(
            FakeTaskEnv(1.0),
            task_index=0,
            num_tasks=2,
        ),
        TaskIdentityObservationEnv(
            FakeTaskEnv(2.0),
            task_index=1,
            num_tasks=2,
        ),
    ]
    env = CanonicalContinualWorldStreamEnv(
        envs,
        ["first-v3", "second-v3"],
        config=CanonicalContinualWorldConfig(
            steps_per_task=1
        ),
    )
    observation, info = env.reset(seed=2)
    torch.testing.assert_close(
        observation,
        torch.tensor([1.0, 1.0, 0.0]),
    )
    assert info["seq_idx"] == 0

    _, _, _, truncated, info = env.step(
        torch.zeros(1)
    )
    assert truncated
    assert info["seq_idx"] == 0
    assert info["TimeLimit.truncated"] is True
    assert env.cur_seq_idx == 1

    observation, info = env.reset(seed=3)
    torch.testing.assert_close(
        observation,
        torch.tensor([2.0, 0.0, 1.0]),
    )
    assert info["seq_idx"] == 1


def test_evaluation_and_matrix_summary() -> None:
    result = evaluate_task(
        ZeroAgent(),
        FakeTaskEnv(1.5),
        episodes=3,
        max_episode_steps=5,
    )
    assert result.mean_return == pytest.approx(3.0)
    assert result.success_rate == pytest.approx(1.0)

    recorder = PerformanceMatrixRecorder(
        ["task-a", "task-b"]
    )
    recorder.append("after-a", [1.0, 0.1])
    recorder.append("after-b", [0.8, 1.0])
    np.testing.assert_allclose(
        recorder.matrix,
        np.array(
            [
                [1.0, 0.1],
                [0.8, 1.0],
            ]
        ),
    )
    summary = recorder.summary()
    assert summary["final_average"] == pytest.approx(0.9)
    assert summary["mean_forgetting"] == pytest.approx(0.2)
    assert summary["backward_transfer"] == pytest.approx(-0.2)
