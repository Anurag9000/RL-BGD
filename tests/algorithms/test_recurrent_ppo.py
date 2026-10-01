import torch

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.recurrent_agent import (
    RecurrentPPOAgent,
    RecurrentPPOConfig,
)
from rl_bgd.agents.ppo.recurrent_train import (
    RecurrentPPOTrainConfig,
    train_recurrent_ppo,
)
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)
from rl_bgd.models.recurrent_ppo import (
    RecurrentValueNetwork,
)


def test_recurrent_value_reset_breaks_pre_reset_history() -> None:
    torch.manual_seed(60)
    model = RecurrentValueNetwork(
        1,
        recurrent_hidden_dim=8,
        encoder_hidden_dims=(8,),
    )
    initial = model.initial_state().squeeze(0)
    starts = torch.tensor(
        [[True], [False], [True], [False]],
        dtype=torch.bool,
    )
    left = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]]
    )
    right = torch.tensor(
        [[-10.0], [20.0], [3.0], [4.0]]
    )
    left_values, _ = model(
        left,
        initial,
        starts,
    )
    right_values, _ = model(
        right,
        initial,
        starts,
    )
    torch.testing.assert_close(
        left_values[2:],
        right_values[2:],
    )


def test_recurrent_ppo_stationary_smoke_is_finite() -> None:
    torch.manual_seed(61)
    env = LinearQuadraticControlEnv(
        horizon=16
    )
    agent = RecurrentPPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        ppo_config=PPOConfig(
            actor_lr=1e-3,
            value_lr=1e-3,
            update_epochs=2,
            minibatch_size=16,
        ),
        recurrent_config=RecurrentPPOConfig(
            recurrent_hidden_dim=16,
            sequence_length=8,
            encoder_hidden_dims=(16,),
        ),
    )
    summary = train_recurrent_ppo(
        env,
        agent,
        config=RecurrentPPOTrainConfig(
            total_steps=96,
            rollout_steps=32,
            seed=61,
        ),
    )
    metrics = summary[
        "last_update_metrics"
    ]
    assert torch.isfinite(
        torch.tensor(
            metrics[
                "policy_loss"
            ]
        )
    )
    assert metrics[
        "sequence_chunks"
    ] > 0
    assert summary[
        "recurrent_reset_count"
    ] >= 1
