import pytest
import torch

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.bgd_agent import BGDPPOConfig
from rl_bgd.agents.ppo.recurrent_agent import RecurrentPPOConfig
from rl_bgd.agents.ppo.recurrent_bgd_agent import BGDRecurrentPPOAgent
from rl_bgd.bayes.bgd import BGDConfig


def make_agent() -> BGDRecurrentPPOAgent:
    return BGDRecurrentPPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        ppo_config=PPOConfig(
            update_epochs=2,
            minibatch_size=4,
        ),
        recurrent_config=RecurrentPPOConfig(
            recurrent_hidden_dim=8,
            sequence_length=4,
            encoder_hidden_dims=(8,),
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


def test_recurrent_bgd_ppo_checkpoint_preserves_posteriors_and_hidden_state() -> None:
    torch.manual_seed(74)
    agent = make_agent()
    agent.sample_action_recurrent(torch.tensor([0.25, -0.5]))
    assert agent.actor_posterior is not None
    assert agent.value_posterior is not None
    first_name = next(iter(agent.actor_posterior.stds))
    agent.actor_posterior.stds[first_name].mul_(0.9)
    state = agent.state_dict()

    restored = make_agent()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.actor_hidden, agent.actor_hidden)
    torch.testing.assert_close(restored.value_hidden, agent.value_hidden)
    assert restored.actor_posterior is not None
    assert restored.value_posterior is not None
    for name in agent.actor_posterior.stds:
        torch.testing.assert_close(
            restored.actor_posterior.stds[name],
            agent.actor_posterior.stds[name],
        )
    for name in agent.value_posterior.stds:
        torch.testing.assert_close(
            restored.value_posterior.stds[name],
            agent.value_posterior.stds[name],
        )


def test_recurrent_bgd_ppo_checkpoint_rejects_bayesian_config_mismatch() -> None:
    agent = make_agent()
    state = agent.state_dict()
    restored = BGDRecurrentPPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        ppo_config=PPOConfig(
            update_epochs=2,
            minibatch_size=4,
        ),
        recurrent_config=RecurrentPPOConfig(
            recurrent_hidden_dim=8,
            sequence_length=4,
            encoder_hidden_dims=(8,),
        ),
        bgd_config=BGDPPOConfig(
            bayesianization="actor_and_value",
            posterior_std=0.2,
            evidence_mode="first_epoch_only",
            actor_bgd=BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
            value_bgd=BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
        ),
    )
    with pytest.raises(
        ValueError,
        match="BGD recurrent PPO checkpoint configuration mismatch",
    ):
        restored.load_state_dict(state)
