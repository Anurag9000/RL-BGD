import torch

from rl_bgd.agents.sac.agent import (
    SACConfig,
)
from rl_bgd.agents.sac.recurrent_agent import (
    RecurrentSACAgent,
    RecurrentSACConfig,
)
from rl_bgd.agents.sac.recurrent_train import (
    RecurrentSACTrainConfig,
    train_recurrent_sac,
)
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)


def test_recurrent_sac_stationary_smoke_is_finite() -> None:
    torch.manual_seed(82)
    env = LinearQuadraticControlEnv(horizon=16)
    agent = RecurrentSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=12,
            encoder_hidden_dims=(12,),
            q_hidden_dims=(12,),
        ),
    )
    summary = train_recurrent_sac(
        env,
        agent,
        config=RecurrentSACTrainConfig(
            total_steps=96,
            random_steps=24,
            sequence_batch_size=4,
            burn_in=2,
            unroll=4,
            replay_capacity=128,
            seed=82,
        ),
    )
    metrics = summary["last_update_metrics"]
    assert metrics
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))
    assert summary["replay_size"] == 96


def test_recurrent_sac_checkpoint_preserves_online_hidden_state() -> None:
    torch.manual_seed(83)
    agent = RecurrentSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=8,
            encoder_hidden_dims=(8,),
            q_hidden_dims=(8,),
        ),
    )
    agent.act_recurrent(torch.tensor([0.2, -0.3]))
    state = agent.state_dict()
    restored = RecurrentSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=8,
            encoder_hidden_dims=(8,),
            q_hidden_dims=(8,),
        ),
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.actor_hidden,
        agent.actor_hidden,
    )


def test_recurrent_sac_post_step_observer_runs_each_step() -> None:
    torch.manual_seed(114)
    env = LinearQuadraticControlEnv(horizon=8)
    agent = RecurrentSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=6,
            encoder_hidden_dims=(6,),
            q_hidden_dims=(6,),
        ),
    )
    observed: list[int] = []
    train_recurrent_sac(
        env,
        agent,
        config=RecurrentSACTrainConfig(
            total_steps=12,
            random_steps=11,
            sequence_batch_size=2,
            burn_in=1,
            unroll=2,
            replay_capacity=16,
            seed=114,
        ),
        post_step_observer=lambda step, _: observed.append(step),
    )
    assert observed == list(range(1, 13))
