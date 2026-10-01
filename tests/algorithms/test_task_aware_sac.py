import math

import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.task_aware_agent import TaskAwareSACAgent
from rl_bgd.models.task_aware_sac import TaskAwareSquashedGaussianActor
from rl_bgd.replay.buffer import ReplayBuffer


def test_task_aware_actor_selects_occurrence_head() -> None:
    actor = TaskAwareSquashedGaussianActor(
        3,
        1,
        num_tasks=2,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(4,),
        use_layer_norm=False,
    )
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.mean_head.bias.copy_(
            torch.tensor([0.0, 1.0])
        )
    first = actor.deterministic(
        torch.tensor([[0.5, 1.0, 0.0]])
    )
    second = actor.deterministic(
        torch.tensor([[0.5, 0.0, 1.0]])
    )
    assert first.item() == 0.0
    assert second.item() > 0.7


def test_task_aware_sac_update_is_finite_and_checkpointable() -> None:
    torch.manual_seed(111)
    agent = TaskAwareSACAgent(
        3,
        1,
        num_tasks=2,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
        config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
    )
    replay = ReplayBuffer(
        64,
        3,
        1,
    )
    for index in range(32):
        task = index % 2
        observation = torch.tensor(
            [
                0.1 * index,
                float(task == 0),
                float(task == 1),
            ]
        )
        replay.add(
            observation,
            torch.tensor([0.0]),
            reward=1.0,
            next_observation=observation.clone(),
            terminated=False,
            truncated=False,
            insertion_step=index,
        )
    batch = replay.sample(
        16,
        generator=torch.Generator().manual_seed(4),
    )
    metrics = agent.update(batch)
    assert all(
        math.isfinite(value)
        for value in metrics.values()
    )

    agent.reset_optimizers()
    state = agent.state_dict()
    restored = TaskAwareSACAgent(
        3,
        1,
        num_tasks=2,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
    )
    restored.load_state_dict(state)
    observation = torch.tensor(
        [0.25, 1.0, 0.0]
    )
    torch.testing.assert_close(
        restored.act(
            observation,
            deterministic=True,
        ),
        agent.act(
            observation,
            deterministic=True,
        ),
    )
    assert restored.optimizer_reset_count == 1
