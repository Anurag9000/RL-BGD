from pathlib import Path

import pytest
import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.recurrent_agent import RecurrentSACAgent, RecurrentSACConfig
from rl_bgd.agents.sac.recurrent_train import (
    RecurrentSACTrainConfig,
    train_recurrent_sac,
)
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.utils.randomness import seed_everything


def _make() -> tuple[LinearQuadraticControlEnv, RecurrentSACAgent]:
    env = LinearQuadraticControlEnv(
        horizon=8,
        process_noise=0.01,
    )
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
            recurrent_hidden_dim=6,
            encoder_hidden_dims=(6,),
            q_hidden_dims=(6,),
        ),
    )
    return env, agent


def test_recurrent_sac_training_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    config = RecurrentSACTrainConfig(
        total_steps=48,
        random_steps=12,
        sequence_batch_size=2,
        burn_in=1,
        unroll=3,
        replay_capacity=64,
        seed=94,
    )

    seed_everything(94, deterministic=True)
    full_env, full_agent = _make()
    full = train_recurrent_sac(
        full_env,
        full_agent,
        config=config,
    )

    checkpoint = tmp_path / "recurrent_sac.pt"
    seed_everything(94, deterministic=True)
    split_env, split_agent = _make()
    partial = train_recurrent_sac(
        split_env,
        split_agent,
        config=config,
        checkpoint_path=checkpoint,
        max_steps_this_call=24,
    )
    assert partial["completed"] is False

    resumed_env, resumed_agent = _make()
    resumed = train_recurrent_sac(
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
    for left, right in zip(
        resumed_agent.actor.parameters(),
        full_agent.actor.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)


def test_recurrent_sac_rejects_corrupt_progress_and_episode_flag(
    tmp_path: Path,
) -> None:
    config = RecurrentSACTrainConfig(
        total_steps=8,
        random_steps=8,
        sequence_batch_size=1,
        burn_in=1,
        unroll=1,
        replay_capacity=8,
        seed=212,
    )
    checkpoint = tmp_path / "valid_recurrent_sac.pt"
    env, agent = _make()
    train_recurrent_sac(
        env,
        agent,
        config=config,
        checkpoint_path=checkpoint,
        max_steps_this_call=6,
    )
    saved = torch.load(checkpoint, weights_only=False)
    for index, (field, value, message) in enumerate(
        [
            ("version", 1.0, "must be an integer"),
            ("next_step", "6", "must be an integer"),
            ("next_step", 5, "replay/step progress mismatch"),
            ("episode_start", 1, "must be a boolean"),
        ]
    ):
        corrupt = dict(saved)
        corrupt[field] = value
        path = tmp_path / f"corrupt_recurrent_sac_{index}.pt"
        torch.save(corrupt, path)
        resumed_env, resumed_agent = _make()
        with pytest.raises((TypeError, ValueError), match=message):
            train_recurrent_sac(
                resumed_env,
                resumed_agent,
                config=config,
                resume_from=path,
            )


def test_recurrent_sac_resume_rolls_back_on_late_environment_failure(
    tmp_path: Path,
) -> None:
    config = RecurrentSACTrainConfig(
        total_steps=8,
        random_steps=8,
        sequence_batch_size=1,
        burn_in=1,
        unroll=1,
        replay_capacity=8,
        seed=321,
    )
    checkpoint = tmp_path / "recurrent_sac_transaction.pt"
    env, agent = _make()
    train_recurrent_sac(
        env,
        agent,
        config=config,
        checkpoint_path=checkpoint,
        max_steps_this_call=6,
    )
    saved = torch.load(checkpoint, weights_only=False)
    environment = dict(saved["environment"])
    environment["step"] = -1
    saved["environment"] = environment
    corrupt = tmp_path / "recurrent_sac_bad_environment.pt"
    torch.save(saved, corrupt)

    target_env, target_agent = _make()
    actor_before = [parameter.detach().clone() for parameter in target_agent.actor.parameters()]
    hidden_before = target_agent.actor_hidden.detach().clone()
    env_before = target_env.state_dict()
    with pytest.raises(ValueError, match="step is invalid"):
        train_recurrent_sac(
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


def test_recurrent_sac_checkpoint_rejects_episode_history_shape_mismatch(
    tmp_path: Path,
) -> None:
    config = RecurrentSACTrainConfig(
        total_steps=8,
        random_steps=8,
        sequence_batch_size=1,
        burn_in=1,
        unroll=1,
        replay_capacity=8,
        seed=402,
    )
    checkpoint = tmp_path / "recurrent_sac_history_shape.pt"
    env, agent = _make()
    train_recurrent_sac(
        env,
        agent,
        config=config,
        checkpoint_path=checkpoint,
        max_steps_this_call=6,
    )
    saved = torch.load(checkpoint, weights_only=False)
    history = list(saved["episode_history"])
    assert history
    history[0] = torch.zeros(2)
    saved["episode_history"] = history
    corrupt = tmp_path / "recurrent_sac_bad_history_shape.pt"
    torch.save(saved, corrupt)

    restored_env, restored_agent = _make()
    with pytest.raises(ValueError, match="shape mismatch"):
        train_recurrent_sac(
            restored_env,
            restored_agent,
            config=config,
            resume_from=corrupt,
        )
