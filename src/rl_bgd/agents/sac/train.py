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
from rl_bgd.utils.checkpoint_io import save_training_checkpoint as _save_training_checkpoint
from rl_bgd.utils.checkpoint_payload import (
    checkpoint_finite_float,
    checkpoint_float_list,
    checkpoint_float_mapping,
    checkpoint_generator_state,
    checkpoint_observation,
)
from rl_bgd.utils.checkpoint_progress import checkpoint_integer, checkpoint_step
from rl_bgd.utils.checkpoint_transaction import transactional_state_load
from rl_bgd.utils.randomness import (
    load_random_state_dict,
    preserved_random_state,
    random_state_dict,
)

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
        if checkpoint_integer(
            checkpoint.get("version"),
            name="SAC training checkpoint version",
        ) != _SAC_TRAINING_CHECKPOINT_VERSION:
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
        environment_state = checkpoint.get("environment")
        if not isinstance(agent_state, dict) or not isinstance(replay_state, dict):
            raise TypeError("SAC training checkpoint agent/replay state must be dictionaries")
        if not isinstance(environment_state, dict):
            raise TypeError("SAC environment checkpoint state must be a dictionary")
        replay_transition_id = checkpoint_integer(
            replay_state.get("next_transition_id"),
            name="SAC replay checkpoint next_transition_id",
        )
        if replay_transition_id != start_step:
            raise ValueError("SAC training checkpoint replay/step progress mismatch")

        observation = checkpoint_observation(
            checkpoint.get("observation"),
            name="SAC training checkpoint observation",
            device=agent.device,
        )
        episode_return = checkpoint_finite_float(
            checkpoint.get("episode_return"),
            name="SAC training checkpoint episode_return",
        )
        completed_returns = checkpoint_float_list(
            checkpoint.get("completed_returns"),
            name="SAC training checkpoint completed_returns",
        )
        last_metrics = checkpoint_float_mapping(
            checkpoint.get("last_metrics"),
            name="SAC training checkpoint last_metrics",
        )
        generator_state = checkpoint_generator_state(
            checkpoint.get("replay_generator_state"),
            name="SAC replay generator checkpoint",
            device=agent.device,
        )
        process_rng = checkpoint.get("process_rng")
        if not isinstance(process_rng, dict):
            raise TypeError("SAC process RNG checkpoint must be a dictionary")
        with preserved_random_state():
            load_random_state_dict(process_rng)

        resume_state = {
            "agent": agent_state,
            "replay": replay_state,
            "environment": environment_state,
            "replay_generator_state": generator_state,
            "process_rng": process_rng,
        }

        def current_resume_state() -> dict[str, Any]:
            return {
                "agent": agent.state_dict(),
                "replay": replay.state_dict(),
                "environment": _environment_state_dict(env),
                "replay_generator_state": generator.get_state().clone(),
                "process_rng": random_state_dict(),
            }

        def apply_resume(payload: dict[str, Any]) -> None:
            agent.load_state_dict(payload["agent"])
            replay.load_state_dict(payload["replay"])
            _load_environment_state(env, payload["environment"])
            generator.set_state(payload["replay_generator_state"].cpu())
            load_random_state_dict(payload["process_rng"])

        transactional_state_load(
            resume_state,
            current_state=current_resume_state,
            apply=apply_resume,
        )

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
