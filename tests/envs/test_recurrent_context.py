import torch

from rl_bgd.envs.recurrent_context import PreviousTransitionContextEnv
from rl_bgd.envs.synthetic.lqr import TensorBox


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
