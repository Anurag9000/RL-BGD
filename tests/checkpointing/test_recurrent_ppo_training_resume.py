from pathlib import Path

import pytest
import torch

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.recurrent_agent import RecurrentPPOAgent, RecurrentPPOConfig
from rl_bgd.agents.ppo.recurrent_train import (
    RecurrentPPOTrainConfig,
    train_recurrent_ppo,
)
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.utils.randomness import seed_everything


def _make() -> tuple[LinearQuadraticControlEnv, RecurrentPPOAgent]:
    env = LinearQuadraticControlEnv(
        horizon=8,
        process_noise=0.01,
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
            minibatch_size=8,
        ),
        recurrent_config=RecurrentPPOConfig(
            recurrent_hidden_dim=6,
            sequence_length=4,
            encoder_hidden_dims=(6,),
        ),
    )
    return env, agent


def test_recurrent_ppo_training_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    config = RecurrentPPOTrainConfig(
        total_steps=64,
        rollout_steps=16,
        seed=95,
    )

    seed_everything(95, deterministic=True)
    full_env, full_agent = _make()
    full = train_recurrent_ppo(
        full_env,
        full_agent,
        config=config,
    )

    checkpoint = tmp_path / "recurrent_ppo.pt"
    seed_everything(95, deterministic=True)
    split_env, split_agent = _make()
    partial = train_recurrent_ppo(
        split_env,
        split_agent,
        config=config,
        checkpoint_path=checkpoint,
        max_rollouts_this_call=2,
    )
    assert partial["completed"] is False

    resumed_env, resumed_agent = _make()
    resumed = train_recurrent_ppo(
        resumed_env,
        resumed_agent,
        config=config,
        resume_from=checkpoint,
    )

    assert resumed["steps"] == full["steps"]
    assert resumed_agent.update_count == full_agent.update_count
    assert resumed_agent.recurrent_reset_count == full_agent.recurrent_reset_count
    torch.testing.assert_close(
        resumed_agent.actor_hidden,
        full_agent.actor_hidden,
    )
    torch.testing.assert_close(
        resumed_agent.value_hidden,
        full_agent.value_hidden,
    )
    for left, right in zip(
        resumed_agent.actor.parameters(),
        full_agent.actor.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)


def test_recurrent_ppo_rejects_invalid_progress_and_episode_flag(
    tmp_path: Path,
) -> None:
    config = RecurrentPPOTrainConfig(total_steps=8, rollout_steps=4, seed=214)
    checkpoint = tmp_path / "valid_recurrent_ppo.pt"
    env, agent = _make()
    train_recurrent_ppo(
        env,
        agent,
        config=config,
        checkpoint_path=checkpoint,
        max_rollouts_this_call=1,
    )
    saved = torch.load(checkpoint, weights_only=False)
    for index, (field, value, message) in enumerate(
        [
            ("version", 1.0, "must be an integer"),
            ("steps", 4.0, "must be an integer"),
            ("rollout_index", 2, "progress is inconsistent"),
            ("steps", 3, "progress is inconsistent"),
            ("episode_start", "False", "must be a boolean"),
        ]
    ):
        corrupt = dict(saved)
        corrupt[field] = value
        path = tmp_path / f"corrupt_recurrent_ppo_{index}.pt"
        torch.save(corrupt, path)
        resumed_env, resumed_agent = _make()
        with pytest.raises((TypeError, ValueError), match=message):
            train_recurrent_ppo(
                resumed_env,
                resumed_agent,
                config=config,
                resume_from=path,
            )



def test_recurrent_ppo_resume_rolls_back_agent_and_environment_on_late_failure(
    tmp_path: Path,
) -> None:
    config = RecurrentPPOTrainConfig(total_steps=8, rollout_steps=4, seed=317)
    checkpoint = tmp_path / "recurrent_ppo_transaction.pt"
    env, agent = _make()
    train_recurrent_ppo(
        env,
        agent,
        config=config,
        checkpoint_path=checkpoint,
        max_rollouts_this_call=1,
    )
    saved = torch.load(checkpoint, weights_only=False)
    environment = dict(saved["environment"])
    environment["step"] = True
    saved["environment"] = environment
    corrupt = tmp_path / "recurrent_ppo_bad_environment.pt"
    torch.save(saved, corrupt)

    target_env, target_agent = _make()
    actor_before = [parameter.detach().clone() for parameter in target_agent.actor.parameters()]
    hidden_before = target_agent.actor_hidden.detach().clone()
    env_before = target_env.state_dict()
    with pytest.raises(ValueError, match="step is invalid"):
        train_recurrent_ppo(
            target_env,
            target_agent,
            config=config,
            resume_from=corrupt,
        )

    for parameter, before in zip(
        target_agent.actor.parameters(),
        actor_before,
        strict=True,
    ):
        torch.testing.assert_close(parameter, before)
    torch.testing.assert_close(target_agent.actor_hidden, hidden_before)
    assert target_env.state_dict()["step"] == env_before["step"]
    torch.testing.assert_close(
        target_env.state_dict()["state"],
        env_before["state"],
    )
