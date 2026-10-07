"""Training and evaluation loops for recurrent sequence-replay SAC."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rl_bgd.agents.sac.recurrent_agent import RecurrentSACAgent
from rl_bgd.envs.protocols import ContinuousTensorEnv
from rl_bgd.replay.sequence_buffer import SequenceReplayBuffer
from rl_bgd.utils.randomness import load_random_state_dict, random_state_dict

PostStepObserver = Callable[[int, RecurrentSACAgent], None]

_RECURRENT_SAC_TRAINING_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class RecurrentSACTrainConfig:
    total_steps: int = 10_000
    random_steps: int = 1_000
    sequence_batch_size: int = 32
    burn_in: int = 8
    unroll: int = 16
    replay_capacity: int = 100_000
    updates_per_step: int = 1
    seed: int = 0

    def validate(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.random_steps < 0:
            raise ValueError("random_steps must be non-negative")
        if (
            self.sequence_batch_size < 1
            or self.burn_in < 0
            or self.unroll < 1
            or self.updates_per_step < 1
        ):
            raise ValueError("invalid recurrent SAC sequence/update configuration")
        if self.replay_capacity < self.burn_in + self.unroll:
            raise ValueError("replay capacity is shorter than one sequence window")


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
        raise TypeError("recurrent SAC training checkpoint must contain a dictionary")
    return payload


def train_recurrent_sac(
    env: ContinuousTensorEnv,
    agent: RecurrentSACAgent,
    *,
    config: RecurrentSACTrainConfig,
    post_step_observer: PostStepObserver | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int | None = None,
    resume_from: str | Path | None = None,
    max_steps_this_call: int | None = None,
) -> dict[str, object]:
    config.validate()
    if checkpoint_interval is not None and checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if checkpoint_interval is not None and checkpoint_path is None:
        raise ValueError("checkpoint_interval requires checkpoint_path")
    if max_steps_this_call is not None and max_steps_this_call < 1:
        raise ValueError("max_steps_this_call must be positive")

    observation_dim = int(env.observation_space.low.numel())
    action_dim = int(env.action_space.low.numel())
    replay = SequenceReplayBuffer(
        config.replay_capacity,
        observation_dim,
        action_dim,
        storage_device=agent.device,
    )
    generator = torch.Generator(device=agent.device).manual_seed(config.seed + 27)
    start_step = 0
    episode_start = True
    episode_history: list[Tensor] = []
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}

    if resume_from is None:
        observation, _ = env.reset(seed=config.seed)
        agent.reset_recurrent_state()
    else:
        checkpoint = _load_training_checkpoint(
            resume_from,
            device=agent.device,
        )
        if checkpoint.get("version") != _RECURRENT_SAC_TRAINING_CHECKPOINT_VERSION:
            raise ValueError("unsupported recurrent SAC training checkpoint version")
        if checkpoint.get("train_config") != asdict(config):
            raise ValueError("recurrent SAC training checkpoint configuration mismatch")
        agent_state = checkpoint.get("agent")
        replay_state = checkpoint.get("replay")
        if not isinstance(agent_state, dict) or not isinstance(replay_state, dict):
            raise TypeError("recurrent SAC agent/replay state must be dictionaries")
        agent.load_state_dict(agent_state)
        replay.load_state_dict(replay_state)
        _load_environment_state(
            env,
            checkpoint.get("environment"),
        )
        observation = checkpoint.get("observation")
        if not isinstance(observation, Tensor):
            raise TypeError("recurrent SAC checkpoint observation must be a tensor")
        observation = observation.to(
            agent.device,
            dtype=torch.float32,
        )
        start_step = int(checkpoint["next_step"])
        if not 0 <= start_step <= config.total_steps:
            raise ValueError("recurrent SAC checkpoint next_step is invalid")
        episode_start = bool(checkpoint["episode_start"])
        raw_history = checkpoint.get("episode_history")
        if not isinstance(raw_history, list) or not all(
            isinstance(item, Tensor) for item in raw_history
        ):
            raise TypeError("recurrent SAC episode history must be a tensor list")
        episode_history = [
            item.to(
                agent.device,
                dtype=torch.float32,
            )
            for item in raw_history
        ]
        episode_return = float(checkpoint["episode_return"])
        raw_returns = checkpoint.get("completed_returns")
        if not isinstance(raw_returns, list):
            raise TypeError("recurrent SAC completed_returns must be a list")
        completed_returns = [float(value) for value in raw_returns]
        raw_metrics = checkpoint.get("last_metrics")
        if not isinstance(raw_metrics, dict):
            raise TypeError("recurrent SAC last_metrics must be a dictionary")
        last_metrics = {str(key): float(value) for key, value in raw_metrics.items()}
        generator_state = checkpoint.get("replay_generator_state")
        if not isinstance(generator_state, Tensor):
            raise TypeError("recurrent SAC replay generator state must be a tensor")
        generator.set_state(generator_state.cpu())
        process_rng = checkpoint.get("process_rng")
        if not isinstance(process_rng, dict):
            raise TypeError("recurrent SAC process RNG state must be a dictionary")
        load_random_state_dict(process_rng)

    window = config.burn_in + config.unroll
    required_size = window + config.sequence_batch_size - 1
    call_end = config.total_steps
    if max_steps_this_call is not None:
        call_end = min(
            config.total_steps,
            start_step + max_steps_this_call,
        )

    def save(next_step: int) -> None:
        if checkpoint_path is None:
            return
        _save_training_checkpoint(
            checkpoint_path,
            {
                "version": _RECURRENT_SAC_TRAINING_CHECKPOINT_VERSION,
                "train_config": asdict(config),
                "next_step": next_step,
                "agent": agent.state_dict(),
                "replay": replay.state_dict(),
                "environment": _environment_state_dict(env),
                "observation": observation.detach().clone(),
                "episode_start": episode_start,
                "episode_history": [
                    item.detach().clone() for item in episode_history
                ],
                "episode_return": episode_return,
                "completed_returns": list(completed_returns),
                "last_metrics": dict(last_metrics),
                "replay_generator_state": generator.get_state().clone(),
                "process_rng": random_state_dict(),
            },
        )

    for step in range(
        start_step,
        call_end,
    ):
        episode_history.append(observation.detach().clone())
        if step < config.random_steps:
            agent.advance_actor_hidden(observation)
            action = env.action_space.sample(generator=generator)
        else:
            action = agent.act_recurrent(
                observation,
                deterministic=False,
            )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)
        replay.add(
            observation,
            action,
            reward,
            next_observation,
            terminated=terminated,
            truncated=truncated,
            episode_start=episode_start,
            insertion_step=step,
        )
        episode_return += reward
        observation = next_observation
        episode_done = terminated or truncated

        if episode_done:
            completed_returns.append(episode_return)
            episode_return = 0.0
            observation, _ = env.reset()
            agent.reset_recurrent_state()
            episode_history = []
            episode_start = True
        else:
            episode_start = False

        updated = False
        if len(replay) >= required_size and step >= config.random_steps:
            for _ in range(config.updates_per_step):
                batch = replay.sample_sequences(
                    config.sequence_batch_size,
                    burn_in=config.burn_in,
                    unroll=config.unroll,
                    generator=generator,
                )
                last_metrics = agent.update(batch)
                updated = True

        if updated and not episode_done:
            agent.rebuild_actor_hidden(episode_history)

        next_step = step + 1
        if post_step_observer is not None:
            post_step_observer(
                next_step,
                agent,
            )
        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and next_step % checkpoint_interval == 0
        ):
            save(next_step)

    if checkpoint_path is not None:
        save(call_end)

    return {
        "steps": call_end,
        "target_steps": config.total_steps,
        "completed": call_end >= config.total_steps,
        "episodes": len(completed_returns),
        "mean_episode_return": (
            sum(completed_returns) / len(completed_returns) if completed_returns else float("nan")
        ),
        "final_10_mean_return": (
            sum(completed_returns[-10:]) / min(10, len(completed_returns))
            if completed_returns
            else float("nan")
        ),
        "last_update_metrics": last_metrics,
        "replay_size": len(replay),
        "recurrent_reset_count": agent.recurrent_reset_count,
    }


@torch.no_grad()
def evaluate_recurrent_sac(
    env: ContinuousTensorEnv,
    agent: RecurrentSACAgent,
    *,
    episodes: int = 5,
    seed: int = 30_000,
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
