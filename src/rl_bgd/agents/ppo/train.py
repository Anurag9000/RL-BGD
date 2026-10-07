"""Small inspectable PPO collection/training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from rl_bgd.agents.ppo.rollout import RolloutBuffer
from rl_bgd.envs.protocols import ContinuousTensorEnv
from rl_bgd.utils.randomness import load_random_state_dict, random_state_dict

_PPO_TRAINING_CHECKPOINT_VERSION = 1


class PPOTrainAgent(Protocol):
    device: torch.device

    def sample_action(
        self,
        observation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def value_of(
        self,
        observation: Tensor,
    ) -> Tensor: ...

    def update(
        self,
        rollout: RolloutBuffer,
    ) -> dict[str, float]: ...

    def act(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None: ...


@dataclass(frozen=True)
class PPOTrainConfig:
    total_steps: int = 10_000
    rollout_steps: int = 1024
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


def _save_training_checkpoint(
    path: str | Path,
    state: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(
            state,
            temporary,
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


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
        raise TypeError("PPO training checkpoint must contain a dictionary")
    return payload


def train_ppo(
    env: ContinuousTensorEnv,
    agent: PPOTrainAgent,
    *,
    config: PPOTrainConfig,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval_rollouts: int | None = None,
    resume_from: str | Path | None = None,
    max_rollouts_this_call: int | None = None,
) -> dict[str, object]:
    if config.total_steps < 1 or config.rollout_steps < 1:
        raise ValueError("PPO training budgets must be positive")
    if checkpoint_interval_rollouts is not None and checkpoint_interval_rollouts < 1:
        raise ValueError("checkpoint_interval_rollouts must be positive")
    if checkpoint_interval_rollouts is not None and checkpoint_path is None:
        raise ValueError("checkpoint_interval_rollouts requires checkpoint_path")
    if max_rollouts_this_call is not None and max_rollouts_this_call < 1:
        raise ValueError("max_rollouts_this_call must be positive")

    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}
    steps = 0
    rollout_index = 0

    if resume_from is None:
        observation, _ = env.reset(seed=config.seed)
    else:
        checkpoint = _load_training_checkpoint(
            resume_from,
            device=agent.device,
        )
        if checkpoint.get("version") != _PPO_TRAINING_CHECKPOINT_VERSION:
            raise ValueError("unsupported PPO training checkpoint version")
        if checkpoint.get("train_config") != asdict(config):
            raise ValueError("PPO training checkpoint configuration mismatch")
        agent_state = checkpoint.get("agent")
        if not isinstance(agent_state, dict):
            raise TypeError("PPO training checkpoint agent state must be a dictionary")
        agent.load_state_dict(agent_state)
        _load_environment_state(
            env,
            checkpoint.get("environment"),
        )
        saved_observation = checkpoint.get("observation")
        if not isinstance(saved_observation, torch.Tensor):
            raise TypeError("PPO training training checkpoint observation must be a tensor")
        observation = saved_observation.to(
            agent.device,
            dtype=torch.float32,
        )
        steps = int(checkpoint["steps"])
        rollout_index = int(checkpoint["rollout_index"])
        if not 0 <= steps <= config.total_steps or rollout_index < 0:
            raise ValueError("PPO training checkpoint progress is invalid")
        episode_return = float(checkpoint["episode_return"])
        completed = checkpoint.get("completed_returns")
        if not isinstance(completed, list):
            raise TypeError("PPO training checkpoint completed_returns must be a list")
        completed_returns = [float(value) for value in completed]
        saved_metrics = checkpoint.get("last_metrics")
        if not isinstance(saved_metrics, dict):
            raise TypeError("PPO training checkpoint last_metrics must be a dictionary")
        last_metrics = {str(key): float(value) for key, value in saved_metrics.items()}
        process_rng = checkpoint.get("process_rng")
        if not isinstance(process_rng, dict):
            raise TypeError("PPO process RNG checkpoint must be a dictionary")
        load_random_state_dict(process_rng)

    def save() -> None:
        if checkpoint_path is None:
            return
        _save_training_checkpoint(
            checkpoint_path,
            {
                "version": _PPO_TRAINING_CHECKPOINT_VERSION,
                "train_config": asdict(config),
                "steps": steps,
                "rollout_index": rollout_index,
                "agent": agent.state_dict(),
                "environment": _environment_state_dict(env),
                "observation": observation.detach().clone(),
                "episode_return": episode_return,
                "completed_returns": list(completed_returns),
                "last_metrics": dict(last_metrics),
                "process_rng": random_state_dict(),
            },
        )

    rollouts_this_call = 0
    while steps < config.total_steps:
        if (
            max_rollouts_this_call is not None
            and rollouts_this_call >= max_rollouts_this_call
        ):
            break

        horizon = min(
            config.rollout_steps,
            config.total_steps - steps,
        )
        observation_dim = int(env.observation_space.low.numel())
        action_dim = int(env.action_space.low.numel())
        rollout = RolloutBuffer(
            horizon,
            observation_dim,
            action_dim,
            device=agent.device,
        )

        for _ in range(horizon):
            (
                action,
                log_prob,
                value,
            ) = agent.sample_action(observation)
            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)
            next_value = agent.value_of(next_observation)
            rollout.add(
                observation,
                action,
                reward,
                terminated=terminated,
                truncated=truncated,
                value=value,
                next_value=next_value,
                log_prob=log_prob,
            )
            episode_return += reward
            observation = next_observation
            steps += 1

            if terminated or truncated:
                completed_returns.append(episode_return)
                episode_return = 0.0
                observation, _ = env.reset()

        last_metrics = agent.update(rollout)
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
    }


@torch.no_grad()
def evaluate_ppo(
    env: ContinuousTensorEnv,
    agent: PPOTrainAgent,
    *,
    episodes: int = 5,
    seed: int = 10_000,
) -> float:
    if episodes < 1:
        raise ValueError("episodes must be positive")

    returns: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        while True:
            observation, reward, terminated, truncated, _ = env.step(
                agent.act(
                    observation,
                    deterministic=True,
                )
            )
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(episode_return)
    return sum(returns) / len(returns)
