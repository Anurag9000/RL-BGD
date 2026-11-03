"""Recurrent CPU replay policy and checkpoint recovery contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.recurrent_agent import RecurrentSACAgent, RecurrentSACConfig
from rl_bgd.agents.sac.recurrent_train import (
    RecurrentSACTrainConfig,
    _replay_storage_device,
    train_recurrent_sac,
)
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.utils.randomness import seed_everything


def make_env_agent(device: str = "cpu") -> tuple[LinearQuadraticControlEnv, RecurrentSACAgent]:
    env = LinearQuadraticControlEnv(horizon=8, process_noise=0.01, device=device)
    agent = RecurrentSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        sac_config=SACConfig(actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3),
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=6, encoder_hidden_dims=(6,), q_hidden_dims=(6,)
        ),
        device=device,
    )
    return env, agent


@pytest.mark.parametrize("storage", ["model", "cpu", "auto"])
def test_cpu_replay_storage_policies_preserve_bounded_resume(tmp_path: Path, storage: str) -> None:
    config = RecurrentSACTrainConfig(
        total_steps=28,
        random_steps=8,
        sequence_batch_size=2,
        burn_in=1,
        unroll=2,
        replay_capacity=40,
        seed=94,
        replay_storage=storage,  # type: ignore[arg-type]
    )
    seed_everything(94, deterministic=True)
    full_env, full_agent = make_env_agent()
    full = train_recurrent_sac(full_env, full_agent, config=config)

    seed_everything(94, deterministic=True)
    split_env, split_agent = make_env_agent()
    checkpoint = tmp_path / f"recurrent_{storage}.pt"
    partial = train_recurrent_sac(
        split_env,
        split_agent,
        config=config,
        checkpoint_path=checkpoint,
        max_steps_this_call=14,
    )
    assert partial["replay_storage_device"] == "cpu"
    assert partial["completed"] is False
    restored_env, restored_agent = make_env_agent()
    resumed = train_recurrent_sac(
        restored_env, restored_agent, config=config, resume_from=checkpoint
    )
    assert resumed["replay_storage_device"] == "cpu"
    assert resumed["steps"] == full["steps"]
    assert restored_agent.update_count == full_agent.update_count
    for a, b in zip(restored_agent.actor.parameters(), full_agent.actor.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_model_policy_reads_legacy_version_one_checkpoint(tmp_path: Path) -> None:
    config = RecurrentSACTrainConfig(
        total_steps=12,
        random_steps=12,
        sequence_batch_size=1,
        burn_in=1,
        unroll=2,
        replay_capacity=16,
        seed=8,
    )
    env, agent = make_env_agent()
    checkpoint_path = tmp_path / "old_recurrent.pt"
    train_recurrent_sac(
        env,
        agent,
        config=config,
        checkpoint_path=checkpoint_path,
        max_steps_this_call=6,
    )
    payload = torch.load(checkpoint_path, weights_only=False)
    payload["version"] = 1
    payload["train_config"].pop("replay_storage")
    payload.pop("sampler_generator_state")
    payload.pop("replay_storage_device")
    torch.save(payload, checkpoint_path)
    restored_env, restored_agent = make_env_agent()
    result = train_recurrent_sac(
        restored_env, restored_agent, config=config, resume_from=checkpoint_path
    )
    assert result["completed"] is True


def test_replay_storage_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    config = RecurrentSACTrainConfig(
        total_steps=10,
        random_steps=10,
        sequence_batch_size=1,
        burn_in=1,
        unroll=1,
        replay_capacity=12,
    )
    env, agent = make_env_agent()
    checkpoint_path = tmp_path / "wrong_device.pt"
    train_recurrent_sac(
        env, agent, config=config, checkpoint_path=checkpoint_path, max_steps_this_call=4
    )
    payload = torch.load(checkpoint_path, weights_only=False)
    payload["replay_storage_device"] = "cuda:1"
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="replay storage device mismatch"):
        env2, agent2 = make_env_agent()
        train_recurrent_sac(env2, agent2, config=config, resume_from=checkpoint_path)


def test_auto_replay_storage_reserves_gpu_memory_without_allocating_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RecurrentSACTrainConfig(
        replay_capacity=10_000_000,
        sequence_batch_size=128,
        burn_in=15,
        unroll=1,
        replay_storage="auto",
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4_000_000_000, 8_000_000_000))
    assert _replay_storage_device(
        torch.device("cuda"), config, observation_dim=45, action_dim=4
    ) == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (32_000_000_000, 64_000_000_000))
    assert _replay_storage_device(
        torch.device("cuda"), config, observation_dim=45, action_dim=4
    ) == torch.device("cuda")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_replay_gpu_model_bounded_update_and_resume(tmp_path: Path) -> None:
    config = RecurrentSACTrainConfig(
        total_steps=18,
        random_steps=6,
        sequence_batch_size=2,
        burn_in=1,
        unroll=2,
        replay_capacity=24,
        replay_storage="cpu",
        seed=34,
    )
    seed_everything(34, deterministic=True)
    env, agent = make_env_agent("cuda")
    checkpoint_path = tmp_path / "cpu_replay_cuda_agent.pt"
    partial = train_recurrent_sac(
        env, agent, config=config, checkpoint_path=checkpoint_path, max_steps_this_call=9
    )
    assert partial["replay_storage_device"] == "cpu"
    assert agent.update_count > 0
    payload = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    assert payload["sampler_generator_state"] is not None
    env2, agent2 = make_env_agent("cuda")
    result = train_recurrent_sac(env2, agent2, config=config, resume_from=checkpoint_path)
    assert result["completed"] is True
