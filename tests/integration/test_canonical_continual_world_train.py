import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.canonical_train import CanonicalSACTrainConfig, train_canonical_sac
from rl_bgd.agents.sac.task_aware_agent import TaskAwareSACAgent
from rl_bgd.envs.continual_world.canonical import (
    CanonicalContinualWorldConfig,
    CanonicalContinualWorldStreamEnv,
    TaskIdentityObservationEnv,
)
from rl_bgd.envs.synthetic.lqr import TensorBox


class TinyTaskEnv:
    def __init__(self, value: float) -> None:
        self.value = value
        self.action_space = TensorBox(
            low=torch.tensor([-1.0]),
            high=torch.tensor([1.0]),
        )
        self.observation_space = TensorBox(
            low=torch.tensor([-5.0]),
            high=torch.tensor([5.0]),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        del seed
        return torch.tensor([self.value]), {}

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, object]]:
        del action
        return torch.tensor([self.value]), self.value, False, False, {"success": 0.0}


def test_canonical_training_resets_replay_and_optimizer_per_task() -> None:
    envs = [
        TaskIdentityObservationEnv(TinyTaskEnv(value), task_index=index, num_tasks=2)
        for index, value in enumerate((1.0, 2.0))
    ]
    env = CanonicalContinualWorldStreamEnv(
        envs,
        ["first-v3", "second-v3"],
        config=CanonicalContinualWorldConfig(steps_per_task=4),
    )
    agent = TaskAwareSACAgent(
        3,
        1,
        num_tasks=2,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(8, 8),
        config=SACConfig(actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3),
    )
    observed_stages: list[int] = []
    summary = train_canonical_sac(
        env,
        agent,
        config=CanonicalSACTrainConfig(
            replay_capacity=16,
            batch_size=2,
            start_steps_per_task=0,
            update_after=2,
            update_every=1,
            seed=3,
        ),
        stage_observer=lambda stage, _: observed_stages.append(stage),
    )
    assert observed_stages == [1, 2]
    assert summary["replay_reset_count"] == 2
    assert summary["optimizer_reset_count"] == 2
    assert summary["stage_replay_sizes"] == [4, 4]
    assert agent.update_count > 0
