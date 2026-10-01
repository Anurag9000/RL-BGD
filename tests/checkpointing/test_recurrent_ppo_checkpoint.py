import torch

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.recurrent_agent import (
    RecurrentPPOAgent,
    RecurrentPPOConfig,
)


def make_agent() -> RecurrentPPOAgent:
    return RecurrentPPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        ppo_config=PPOConfig(
            update_epochs=1,
            minibatch_size=4,
        ),
        recurrent_config=RecurrentPPOConfig(
            recurrent_hidden_dim=8,
            sequence_length=4,
            encoder_hidden_dims=(8,),
        ),
    )


def test_recurrent_ppo_checkpoint_preserves_hidden_state() -> None:
    torch.manual_seed(62)
    agent = make_agent()
    agent.sample_action_recurrent(torch.tensor([0.5, -0.5]))
    state = agent.state_dict()
    restored = make_agent()
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.actor_hidden,
        agent.actor_hidden,
    )
    torch.testing.assert_close(
        restored.value_hidden,
        agent.value_hidden,
    )
    assert restored.recurrent_reset_count == agent.recurrent_reset_count
