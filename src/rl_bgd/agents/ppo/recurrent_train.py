"""Collection and evaluation loops for recurrent PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rl_bgd.agents.ppo.recurrent_agent import RecurrentPPOAgent
from rl_bgd.agents.ppo.recurrent_rollout import RecurrentRolloutBuffer
from rl_bgd.envs.protocols import ContinuousTensorEnv
from rl_bgd.utils.checkpoint_io import save_training_checkpoint as _save_training_checkpoint
from rl_bgd.utils.checkpoint_payload import (
    checkpoint_finite_float,
    checkpoint_float_list,
    checkpoint_float_mapping,
    checkpoint_observation,
)
from rl_bgd.utils.checkpoint_progress import (
    checkpoint_boolean,
    checkpoint_integer,
    checkpoint_ppo_progress,
)
from rl_bgd.utils.checkpoint_transaction import transactional_state_load
from rl_bgd.utils.randomness import (
    load_random_state_dict,
    preserved_random_state,
    random_state_dict,
)

_RECURRENT_PPO_TRAINING_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class RecurrentPPOTrainConfig:
    total_steps: int = 10_000
    rollout_steps: int = 256
    seed: int = 0


def _environment_state_dict(env: ContinuousTensorEnv) -> dict[str, Any]:
    state_fn = getattr(env, "state_dict", None)
    if not callable(state_fn):
        raise TypeError("environment does not support checkpointing")
    state = state_fn()
    if not isinstance(state, dict):
        raise TypeError("environment state_dict must return a dictionary")
    return state


def _load_environment_state(
    env: ContinuousTensorEnv,
    state: object,
) -> None:
    load_fn = getattr(env, "load_state_dict", None)
    if not callable(load_fn):
        raise TypeError("environment does not support checkpoint restore")
    if not isinstance(state, dict):
        raise TypeError("environment checkpoint state must be a dictionary")
    load_fn(state)


def _load_training_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise TypeError("recurrent PPO training checkpoint must contain a dictionary")
    return payload


def train_recurrent_ppo(
    env: ContinuousTensorEnv,
    agent: RecurrentPPOAgent,
    *,
    config: RecurrentPPOTrainConfig,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval_rollouts: int | None = None,
    resume_from: str | Path | None = None,
    max_rollouts_this_call: int | None = None,
) -> dict[str, object]:
    if config.total_steps < 1 or config.rollout_steps < 1:
        raise ValueError("recurrent PPO training budgets must be positive")
    if checkpoint_interval_rollouts is not None and checkpoint_interval_rollouts < 1:
        raise ValueError("checkpoint_interval_rollouts must be positive")
    if checkpoint_interval_rollouts is not None and checkpoint_path is None:
        raise ValueError("checkpoint_interval_rollouts requires checkpoint_path")
    if max_rollouts_this_call is not None and max_rollouts_this_call < 1:
        raise ValueError("max_rollouts_this_call must be positive")

    episode_start = True
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}
    steps = 0
    rollout_index = 0

    if resume_from is None:
        observation, _ = env.reset(seed=config.seed)
        agent.reset_recurrent_state()
    else:
        checkpoint = _load_training_checkpoint(
            resume_from,
            device=agent.device,
        )
        if checkpoint_integer(
            checkpoint.get("version"),
            name="recurrent PPO training checkpoint version",
        ) != _RECURRENT_PPO_TRAINING_CHECKPOINT_VERSION:
            raise ValueError("unsupported recurrent PPO training checkpoint version")
        if checkpoint.get("train_config") != asdict(config):
            raise ValueError("recurrent PPO training checkpoint configuration mismatch")
        steps, rollout_index = checkpoint_ppo_progress(
            checkpoint,
            total_steps=config.total_steps,
            rollout_steps=config.rollout_steps,
            label="recurrent PPO checkpoint",
        )
        episode_start = checkpoint_boolean(
            checkpoint.get("episode_start"),
            name="recurrent PPO checkpoint episode_start",
        )

        agent_state = checkpoint.get("agent")
        if not isinstance(agent_state, dict):
            raise TypeError("recurrent PPO checkpoint agent state must be a dictionary")
        environment_state = checkpoint.get("environment")
        if not isinstance(environment_state, dict):
            raise TypeError("recurrent PPO environment state must be a dictionary")
        observation = checkpoint_observation(
            checkpoint.get("observation"),
            name="recurrent PPO checkpoint observation",
            device=agent.device,
        )
        episode_return = checkpoint_finite_float(
            checkpoint.get("episode_return"),
            name="recurrent PPO checkpoint episode_return",
        )
        completed_returns = checkpoint_float_list(
            checkpoint.get("completed_returns"),
            name="recurrent PPO completed_returns",
        )
        last_metrics = checkpoint_float_mapping(
            checkpoint.get("last_metrics"),
            name="recurrent PPO last_metrics",
        )
        process_rng = checkpoint.get("process_rng")
        if not isinstance(process_rng, dict):
            raise TypeError("recurrent PPO process RNG state must be a dictionary")
        with preserved_random_state():
            load_random_state_dict(process_rng)

        resume_state = {
            "agent": agent_state,
            "environment": environment_state,
            "process_rng": process_rng,
        }

        def current_resume_state() -> dict[str, Any]:
            return {
                "agent": agent.state_dict(),
                "environment": _environment_state_dict(env),
                "process_rng": random_state_dict(),
            }

        def apply_resume(payload: dict[str, Any]) -> None:
            agent.load_state_dict(payload["agent"])
            _load_environment_state(env, payload["environment"])
            load_random_state_dict(payload["process_rng"])

        transactional_state_load(
            resume_state,
            current_state=current_resume_state,
            apply=apply_resume,
        )

    def save() -> None:
        if checkpoint_path is None:
            return
        _save_training_checkpoint(
            checkpoint_path,
            {
                "version": _RECURRENT_PPO_TRAINING_CHECKPOINT_VERSION,
                "train_config": asdict(config),
                "steps": steps,
                "rollout_index": rollout_index,
                "agent": agent.state_dict(),
                "environment": _environment_state_dict(env),
                "observation": observation.detach().clone(),
                "episode_start": episode_start,
                "episode_return": episode_return,
                "completed_returns": list(completed_returns),
                "last_metrics": dict(last_metrics),
                "process_rng": random_state_dict(),
            },
        )

    rollouts_this_call = 0
    while steps < config.total_steps:
        if max_rollouts_this_call is not None and rollouts_this_call >= max_rollouts_this_call:
            break

        horizon = min(
            config.rollout_steps,
            config.total_steps - steps,
        )
        rollout = RecurrentRolloutBuffer(
            horizon,
            int(env.observation_space.low.numel()),
            int(env.action_space.low.numel()),
            agent.recurrent_config.recurrent_hidden_dim,
            agent.recurrent_config.recurrent_hidden_dim,
            device=agent.device,
        )
        last_transition_done = False
        for _ in range(horizon):
            action_step = agent.sample_action_recurrent(observation)
            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action_step.action)
            next_value = agent.value_from_hidden(
                next_observation,
                action_step.value_hidden_after,
            )
            rollout.add(
                observation,
                action_step.action,
                reward,
                terminated=terminated,
                truncated=truncated,
                value=action_step.value,
                next_value=next_value,
                log_prob=action_step.log_prob,
                episode_start=episode_start,
                actor_hidden=action_step.actor_hidden_before,
                value_hidden=action_step.value_hidden_before,
            )
            episode_return += reward
            observation = next_observation
            steps += 1
            last_transition_done = terminated or truncated
            if last_transition_done:
                completed_returns.append(episode_return)
                episode_return = 0.0
                observation, _ = env.reset()
                agent.reset_recurrent_state()
                episode_start = True
            else:
                episode_start = False

        last_metrics = agent.update(rollout)
        if not last_transition_done:
            agent.refresh_recurrent_state(rollout)
        rollout_index += 1
        rollouts_this_call += 1
        if (
            checkpoint_path is not None
            and checkpoint_interval_rollouts is not None
            and rollout_index % checkpoint_interval_rollouts == 0
        ):
            save()

    if checkpoint_path is not None:
        save()

    return {
        "steps": steps,
        "target_steps": config.total_steps,
        "completed": steps >= config.total_steps,
        "rollouts": rollout_index,
        "episodes": len(completed_returns),
        "mean_episode_return": (
            sum(completed_returns) / len(completed_returns) if completed_returns else float("nan")
        ),
        "final_10_mean_return": (
            sum(completed_returns[-10:])
            / min(
                10,
                len(completed_returns),
            )
            if completed_returns
            else float("nan")
        ),
        "last_update_metrics": last_metrics,
        "recurrent_reset_count": agent.recurrent_reset_count,
    }


@torch.no_grad()
def evaluate_recurrent_ppo(
    env: ContinuousTensorEnv,
    agent: RecurrentPPOAgent,
    *,
    episodes: int = 5,
    seed: int = 20_000,
) -> float:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    returns: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        agent.reset_recurrent_state()
        episode_return = 0.0
        while True:
            action = agent.act_recurrent(
                observation,
                deterministic=True,
            )
            (
                observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(episode_return)
    return sum(returns) / len(returns)
