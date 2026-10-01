import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.task_aware_agent import TaskAwareSACAgent
from rl_bgd.agents.sac.task_aware_train import (
    CanonicalSACTrainConfig,
    train_canonical_task_aware_sac,
)
from rl_bgd.envs.continual_world.canonical import (
    CanonicalContinualWorldConfig,
    CanonicalContinualWorldStreamEnv,
    TaskIdentityObservationEnv,
)
from rl_bgd.envs.synthetic.lqr import TensorBox


class TinyTask:
    def __init__(self, value: float) -> None:
        self.value = value
        self.action_space = TensorBox(
            low=torch.tensor([-1.0]),
            high=torch.tensor([1.0]),
        )
        self.observation_space = TensorBox(
            low=torch.tensor([-2.0]),
            high=torch.tensor([2.0]),
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
        return torch.tensor([self.value]), self.value, False, False, {}


def test_canonical_trainer_resets_replay_optimizer_and_task_clock() -> None:
    env = CanonicalContinualWorldStreamEnv(
        [
            TaskIdentityObservationEnv(TinyTask(0.5), task_index=0, num_tasks=2),
            TaskIdentityObservationEnv(TinyTask(1.0), task_index=1, num_tasks=2),
        ],
        ["task-a", "task-b"],
        config=CanonicalContinualWorldConfig(steps_per_task=2),
    )
    agent = TaskAwareSACAgent(
        3,
        1,
        num_tasks=2,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(8,),
        config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
    )
    stage_steps: list[int] = []

    def observe_stage(
        completed_steps: int,
        current_agent: TaskAwareSACAgent,
    ) -> None:
        del current_agent
        if completed_steps % 2 == 0:
            stage_steps.append(completed_steps)

    result = train_canonical_task_aware_sac(
        env,
        agent,
        config=CanonicalSACTrainConfig(
            total_steps=4,
            start_steps=0,
            update_after=0,
            update_every=1,
            batch_size=1,
            replay_capacity=8,
            seed=5,
        ),
        stage_observer=observe_stage,
    )
    assert result["steps"] == 4
    assert result["episodes"] == 2
    assert result["replay_size"] == 2
    assert result["optimizer_resets"] == 2
    assert agent.update_count == 4
    assert stage_steps == [2, 4]
