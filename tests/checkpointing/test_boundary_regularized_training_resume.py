"""Exact resume semantics for oracle-boundary regularized SAC training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.regularized_agent import (
    RegularizedSACAgent,
    RegularizedSACConfig,
)
from rl_bgd.agents.sac.regularized_train import (
    BoundaryRegularizedSACTrainConfig,
    train_boundary_regularized_sac,
)
from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.envs.synthetic.nonstationary_lqr import ScheduledLQREnv
from rl_bgd.utils.randomness import seed_everything


def _build(
    seed: int,
) -> tuple[
    ScheduledLQREnv,
    RegularizedSACAgent,
    BoundaryRegularizedSACTrainConfig,
]:
    seed_everything(seed, deterministic=True)
    env = ScheduledLQREnv(
        LinearQuadraticControlEnv(horizon=8),
        ContextSchedule(
            ContextScheduleConfig(
                mode="recurring",
                anchors=(
                    {"dynamics": 0.75, "control_gain": 0.35},
                    {"dynamics": 0.95, "control_gain": 0.65},
                ),
                phase_steps=12,
                seed=seed,
            )
        ),
    )
    agent = RegularizedSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(8,),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        regularization_config=RegularizedSACConfig(
            method="ewc",
            target="actor_and_critic",
            strength=0.1,
            consolidation_interval_updates=1_000_000,
            importance_samples=2,
        ),
    )
    config = BoundaryRegularizedSACTrainConfig(
        total_steps=24,
        consolidation_steps=(12,),
        random_steps=4,
        batch_size=4,
        replay_capacity=32,
        importance_batch_size=4,
        seed=seed,
    )
    return env, agent, config


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _assert_nested_equal(lhs, rhs)
        return
    assert left == right


def test_boundary_regularized_training_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    full_env, full_agent, config = _build(123)
    full_result = train_boundary_regularized_sac(
        full_env,
        full_agent,
        config=config,
    )
    full_state = deepcopy(full_agent.state_dict())

    split_env, split_agent, split_config = _build(123)
    checkpoint = tmp_path / "boundary_regularized.pt"
    partial = train_boundary_regularized_sac(
        split_env,
        split_agent,
        config=split_config,
        checkpoint_path=checkpoint,
        max_steps_this_call=9,
    )
    assert partial["completed"] is False
    assert partial["steps"] == 9

    resumed_env, resumed_agent, resumed_config = _build(123)
    seed_everything(999, deterministic=True)
    resumed = train_boundary_regularized_sac(
        resumed_env,
        resumed_agent,
        config=resumed_config,
        checkpoint_path=checkpoint,
        resume_from=checkpoint,
    )

    assert resumed["completed"] is True
    assert resumed["steps"] == 24
    assert resumed["consolidations"] == full_result["consolidations"]
    assert resumed["last_update_metrics"] == full_result["last_update_metrics"]
    _assert_nested_equal(resumed_agent.state_dict(), full_state)


def test_boundary_regularized_resume_rejects_phase_replay_progress_mismatch(
    tmp_path: Path,
) -> None:
    env, agent, config = _build(321)
    checkpoint = tmp_path / "boundary_regularized.pt"
    train_boundary_regularized_sac(
        env,
        agent,
        config=config,
        checkpoint_path=checkpoint,
        max_steps_this_call=7,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    phase_replay = payload["phase_replay"]
    assert isinstance(phase_replay, dict)
    phase_replay["next_transition_id"] = 6
    torch.save(payload, checkpoint)

    target_env, target_agent, target_config = _build(321)
    before = deepcopy(target_agent.state_dict())
    with pytest.raises(ValueError, match="phase replay/step progress mismatch"):
        train_boundary_regularized_sac(
            target_env,
            target_agent,
            config=target_config,
            resume_from=checkpoint,
        )
    _assert_nested_equal(target_agent.state_dict(), before)


@pytest.mark.parametrize(
    ("buffer_name", "resume_step"),
    [("replay", 7), ("phase_replay", 7), ("phase_replay", 16)],
)
def test_boundary_regularized_resume_rejects_insertion_clock(
    tmp_path: Path,
    buffer_name: str,
    resume_step: int,
) -> None:
    env, agent, config = _build(721)
    checkpoint = tmp_path / "invalid_clock.pt"
    train_boundary_regularized_sac(
        env,
        agent,
        config=config,
        checkpoint_path=checkpoint,
        max_steps_this_call=resume_step,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload[buffer_name]
    timestamps = saved["insertion_steps"]
    assert isinstance(timestamps, torch.Tensor) and timestamps.numel() > 0
    altered = timestamps.clone()
    altered[0, 0] += 1
    saved["insertion_steps"] = altered
    torch.save(payload, checkpoint)

    target_env, target_agent, target_config = _build(721)
    before_agent = deepcopy(target_agent.state_dict())
    before_env = deepcopy(target_env.state_dict())
    with pytest.raises(ValueError, match="insertion-step chronology mismatch"):
        train_boundary_regularized_sac(
            target_env,
            target_agent,
            config=target_config,
            resume_from=checkpoint,
        )
    _assert_nested_equal(target_agent.state_dict(), before_agent)
    _assert_nested_equal(target_env.state_dict(), before_env)


@pytest.mark.parametrize("buffer_name", ["replay", "phase_replay"])
def test_boundary_regularized_resume_rejects_dropped_history(
    tmp_path: Path,
    buffer_name: str,
) -> None:
    env, agent, config = _build(722)
    checkpoint = tmp_path / "invalid_size.pt"
    train_boundary_regularized_sac(
        env,
        agent,
        config=config,
        checkpoint_path=checkpoint,
        max_steps_this_call=7,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload[buffer_name]["size"] = 6
    torch.save(payload, checkpoint)

    target_env, target_agent, target_config = _build(722)
    before = deepcopy(target_agent.state_dict())
    with pytest.raises(ValueError, match="size disagrees with checkpoint progress"):
        train_boundary_regularized_sac(
            target_env,
            target_agent,
            config=target_config,
            resume_from=checkpoint,
        )
    _assert_nested_equal(target_agent.state_dict(), before)
