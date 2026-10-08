"""Small inspectable SAC collection/training loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rl_bgd.agents.sac.agent import SACAgent
from rl_bgd.envs.protocols import ContinuousTensorEnv
from rl_bgd.replay.buffer import ReplayBuffer
from rl_bgd.utils.checkpoint_progress import checkpoint_step
from rl_bgd.utils.randomness import load_random_state_dict, random_state_dict

UpdateObserver = Callable[[int, dict[str, float]], None]
EpisodeObserver = Callable[[int, float], None]
PostStepObserver = Callable[[int, SACAgent], None]

_SAC_TRAINING_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class SACTrainConfig:
    total_steps: int = 10_000
    random_steps: int = 1_000
    batch_size: int = 256
    replay_capacity: int = 100_000
    updates_per_step: int = 1
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
        raise TypeError("SAC training checkpoint must contain a dictionary")
    return payload


def train_sac(
    env: ContinuousTensorEnv,
    agent: SACAgent,
    *,
    config: SACTrainConfig,
    update_observer: UpdateObserver | None = None,
    episode_observer: EpisodeObserver | None = None,
    post_step_observer: PostStepObserver | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int | None = None,
    resume_from: str | Path | None = None,
    max_steps_this_call: int | None = None,
) -> dict[str, object]:
    if (
        config.total_steps < 1
        or config.batch_size < 1
        or config.replay_capacity < config.batch_size
    ):
        raise ValueError("invalid SAC training budget/replay configuration")
    if config.random_steps < 0 or config.updates_per_step < 1:
        raise ValueError("invalid SAC warmup/update configuration")
    if checkpoint_interval is not None and checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if checkpoint_interval is not None and checkpoint_path is None:
        raise ValueError("checkpoint_interval requires checkpoint_path")
    if max_steps_this_call is not None and max_steps_this_call < 1:
        raise ValueError("max_steps_this_call must be positive")

    action_dim = int(env.action_space.low.numel())
    observation_dim = int(env.observation_space.low.numel())
    replay = ReplayBuffer(
        config.replay_capacity,
        observation_dim,
        action_dim,
        storage_device=agent.device,
    )
    generator = torch.Generator(device=agent.device).manual_seed(config.seed + 17)

    start_step = 0
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}

    if resume_from is None:
        observation, _ = env.reset(seed=config.seed)
    else:
        checkpoint = _load_training_checkpoint(
            resume_from,
            device=agent.device,
        )
        if checkpoint.get("version") != _SAC_TRAINING_CHECKPOINT_VERSION:
            raise ValueError("unsupported SAC training checkpoint version")
        if checkpoint.get("train_config") != asdict(config):
            raise ValueError("SAC training checkpoint configuration mismatch")
        start_step = checkpoint_step(
            checkpoint.get("next_step"),
            name="SAC training checkpoint next_step",
            limit=config.total_steps,
        )
        agent_state = checkpoint.get("agent")
        replay_state = checkpoint.get("replay")
        if not isinstance(agent_state, dict) or not isinstance(replay_state, dict):
            raise TypeError("SAC training checkpoint agent/replay state must be dictionaries")
        if replay_state.get("next_transition_id") != start_step:
            raise ValueError("SAC training checkpoint replay/step progress mismatch")
        agent.load_state_dict(agent_state)
        replay.load_state_dict(replay_state)
        _load_environment_state(
            env,
            checkpoint.get("environment"),
        )
        saved_observation = checkpoint.get("observation")
        if not isinstance(saved_observation, torch.Tensor):
            raise TypeError("SAC training training checkpoint observation must be a tensor")
        observation = saved_observation.to(
            agent.device,
            dtype=torch.float32,
        )
        episode_return = float(checkpoint["episode_return"])
        completed = checkpoint.get("completed_returns")
        if not isinstance(completed, list):
            raise TypeError("SAC training checkpoint completed_returns must be a list")
        completed_returns = [float(value) for value in completed]
        saved_metrics = checkpoint.get("last_metrics")
        if not isinstance(saved_metrics, dict):
            raise TypeError("SAC training checkpoint last_metrics must be a dictionary")
        last_metrics = {str(key): float(value) for key, value in saved_metrics.items()}
        generator_state = checkpoint.get("replay_generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("SAC replay generator checkpoint must be a tensor")
        generator.set_state(generator_state.cpu())
        process_rng = checkpoint.get("process_rng")
        if not isinstance(process_rng, dict):
            raise TypeError("SAC process RNG checkpoint must be a dictionary")
        load_random_state_dict(process_rng)

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
                "version": _SAC_TRAINING_CHECKPOINT_VERSION,
                "train_config": asdict(config),
                "next_step": next_step,
                "agent": agent.state_dict(),
                "replay": replay.state_dict(),
                "environment": _environment_state_dict(env),
                "observation": observation.detach().clone(),
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
        if step < config.random_steps:
            action = env.action_space.sample(generator=generator)
        else:
            action = agent.act(
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
            insertion_step=step,
        )
        episode_return += reward
        observation = next_observation
        if terminated or truncated:
            completed_returns.append(episode_return)
            if episode_observer is not None:
                episode_observer(
                    step,
                    episode_return,
                )
            episode_return = 0.0
            observation, _ = env.reset()

        if len(replay) >= config.batch_size and step >= config.random_steps:
            for _ in range(config.updates_per_step):
                batch = replay.sample(
                    config.batch_size,
                    generator=generator,
                )
                last_metrics = agent.update(batch)
                if update_observer is not None:
                    update_observer(
                        step,
                        dict(last_metrics),
                    )

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
    }
