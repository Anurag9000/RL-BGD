import torch

from rl_bgd.agents.ppo.agent import (
    PPOConfig,
)
from rl_bgd.agents.ppo.bgd_agent import (
    BGDPPOConfig,
)
from rl_bgd.agents.ppo.recurrent_agent import (
    RecurrentPPOConfig,
)
from rl_bgd.agents.ppo.recurrent_bgd_agent import (
    BGDRecurrentPPOAgent,
)
from rl_bgd.agents.ppo.recurrent_train import (
    RecurrentPPOTrainConfig,
    train_recurrent_ppo,
)
from rl_bgd.bayes.bgd import (
    BGDConfig,
)
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)


def test_bgd_recurrent_ppo_smoke_updates_posteriors() -> None:
    torch.manual_seed(72)
    env = LinearQuadraticControlEnv(
        horizon=16
    )
    agent = BGDRecurrentPPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        ppo_config=PPOConfig(
            update_epochs=2,
            minibatch_size=16,
        ),
        recurrent_config=RecurrentPPOConfig(
            recurrent_hidden_dim=12,
            sequence_length=8,
            encoder_hidden_dims=(12,),
        ),
        bgd_config=BGDPPOConfig(
            bayesianization="actor_and_value",
            evidence_mode="first_epoch_only",
            actor_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
            value_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
        ),
    )
    summary = train_recurrent_ppo(
        env,
        agent,
        config=RecurrentPPOTrainConfig(
            total_steps=64,
            rollout_steps=32,
            seed=72,
        ),
    )
    metrics = summary[
        "last_update_metrics"
    ]
    assert metrics[
        "actor_sigma_mean"
    ] > 0.0
    assert metrics[
        "value_sigma_mean"
    ] > 0.0
    assert (
        metrics[
            "uncertainty_evidence_weight_mean"
        ]
        == 0.5
    )
