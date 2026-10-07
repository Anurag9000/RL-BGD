import pytest
import torch

from rl_bgd.envs.recurrent_context import PreviousTransitionContextEnv
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv, TensorBox


class TinyEnv:
    def __init__(self) -> None:
        self.action_space = TensorBox(
            low=torch.tensor([-2.0]),
            high=torch.tensor([2.0]),
        )
        self.observation_space = TensorBox(
            low=torch.tensor([-5.0, -5.0]),
            high=torch.tensor([5.0, 5.0]),
        )
        self.close_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        del seed
        return torch.tensor([1.0, -1.0]), {"source": "reset"}

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, object]]:
        return (
            torch.tensor([0.25, 0.5]),
            3.5,
            False,
            True,
            {"source": "step"},
        )

    def close(self) -> None:
        self.close_count += 1


def test_previous_transition_context_is_exact_and_task_agnostic() -> None:
    base = TinyEnv()
    env = PreviousTransitionContextEnv(base)

    observation, info = env.reset(seed=3)
    torch.testing.assert_close(
        observation,
        torch.tensor([1.0, -1.0, 0.0, 0.0, 1.0]),
    )
    assert info == {"source": "reset"}
    assert env.context_layout == {
        "observation": 2,
        "previous_action": 1,
        "previous_reward": 1,
        "previous_done": 1,
    }

    action = torch.tensor([0.75])
    next_observation, reward, terminated, truncated, info = env.step(action)
    torch.testing.assert_close(
        next_observation,
        torch.tensor([0.25, 0.5, 0.75, 3.5, 1.0]),
    )
    assert reward == 3.5
    assert not terminated
    assert truncated
    assert info == {"source": "step"}

    env.close()
    assert base.close_count == 1


def test_previous_transition_context_checkpoint_rejects_reward_bound_mismatch() -> None:
    source = PreviousTransitionContextEnv(
        LinearQuadraticControlEnv(),
        reward_bound=10.0,
    )
    source.reset(seed=7)
    state = source.state_dict()

    restored = PreviousTransitionContextEnv(
        LinearQuadraticControlEnv(),
        reward_bound=20.0,
    )
    with pytest.raises(ValueError, match="reward bound mismatch"):
        restored.load_state_dict(state)


def test_previous_transition_context_checkpoint_round_trip() -> None:
    source = PreviousTransitionContextEnv(
        LinearQuadraticControlEnv(process_noise=0.05),
        reward_bound=10.0,
    )
    source.reset(seed=8)
    source.step(torch.tensor([0.25]))
    state = source.state_dict()
    expected = source.step(torch.tensor([-0.5]))

    restored = PreviousTransitionContextEnv(
        LinearQuadraticControlEnv(process_noise=0.05),
        reward_bound=10.0,
    )
    restored.load_state_dict(state)
    actual = restored.step(torch.tensor([-0.5]))

    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1:] == expected[1:]
