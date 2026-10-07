import pytest
import torch

from rl_bgd.agents.sac.agent import (
    SACConfig,
)
from rl_bgd.agents.sac.bgd_agent import (
    BGDSACConfig,
)
from rl_bgd.agents.sac.recurrent_agent import (
    RecurrentSACConfig,
)
from rl_bgd.agents.sac.recurrent_bgd_agent import (
    BGDRecurrentSACAgent,
)
from rl_bgd.agents.sac.recurrent_train import (
    RecurrentSACTrainConfig,
    train_recurrent_sac,
)
from rl_bgd.bayes.bgd import (
    BGDConfig,
)
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)
from rl_bgd.replay.evidence_accounting import (
    ReplayEvidenceConfig,
)


@pytest.mark.parametrize(
    "mode",
    [
        "critic_only",
        "actor_only",
        "actor_and_critic",
    ],
)
def test_bgd_recurrent_sac_modes_are_finite(
    mode: str,
) -> None:
    torch.manual_seed(84)
    env = LinearQuadraticControlEnv(horizon=16)
    agent = BGDRecurrentSACAgent(
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
            recurrent_hidden_dim=8,
            encoder_hidden_dims=(8,),
            q_hidden_dims=(8,),
        ),
        bgd_config=BGDSACConfig(
            bayesianization=mode,  # type: ignore[arg-type]
            posterior_std=0.1,
            replay_evidence=ReplayEvidenceConfig(mode="inverse_reuse_weight"),
            actor_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
            critic_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
        ),
    )
    summary = train_recurrent_sac(
        env,
        agent,
        config=RecurrentSACTrainConfig(
            total_steps=64,
            random_steps=20,
            sequence_batch_size=3,
            burn_in=2,
            unroll=4,
            replay_capacity=96,
            seed=84,
        ),
    )
    metrics = summary["last_update_metrics"]
    assert metrics
    assert torch.isfinite(torch.tensor(metrics["actor_loss"]))
    assert metrics["evidence_weight_mean"] > 0.0


def test_bgd_recurrent_sac_checkpoint_round_trip() -> None:
    torch.manual_seed(85)
    config = BGDSACConfig(
        bayesianization="actor_and_critic",
        actor_bgd=BGDConfig(
            eta=0.1,
            mc_samples=2,
            antithetic=True,
        ),
        critic_bgd=BGDConfig(
            eta=0.1,
            mc_samples=2,
            antithetic=True,
        ),
    )
    recurrent = RecurrentSACConfig(
        recurrent_hidden_dim=8,
        encoder_hidden_dims=(8,),
        q_hidden_dims=(8,),
    )
    agent = BGDRecurrentSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=recurrent,
        bgd_config=config,
    )
    agent.act_recurrent(torch.tensor([0.2, -0.4]))
    state = agent.state_dict()
    restored = BGDRecurrentSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=recurrent,
        bgd_config=config,
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.actor_hidden,
        agent.actor_hidden,
    )
    assert restored.actor_posterior is not None
    assert agent.actor_posterior is not None
    for name in agent.actor_posterior.stds:
        torch.testing.assert_close(
            restored.actor_posterior.stds[name],
            agent.actor_posterior.stds[name],
        )


def test_bgd_recurrent_sac_checkpoint_rejects_bayesian_config_mismatch() -> None:
    recurrent = RecurrentSACConfig(
        recurrent_hidden_dim=8,
        encoder_hidden_dims=(8,),
        q_hidden_dims=(8,),
    )
    base = BGDSACConfig(
        bayesianization="critic_only",
        posterior_std=0.1,
        critic_bgd=BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
    )
    state = BGDRecurrentSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=recurrent,
        bgd_config=base,
    ).state_dict()
    restored = BGDRecurrentSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=recurrent,
        bgd_config=BGDSACConfig(
            bayesianization="critic_only",
            posterior_std=0.2,
            critic_bgd=BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
        ),
    )
    with pytest.raises(ValueError, match="BGD recurrent SAC checkpoint configuration mismatch"):
        restored.load_state_dict(state)
