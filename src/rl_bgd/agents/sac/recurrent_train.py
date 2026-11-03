"""Training and evaluation loops for recurrent sequence-replay SAC."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from rl_bgd.agents.sac.recurrent_agent import RecurrentSACAgent
from rl_bgd.envs.protocols import ContinuousTensorEnv
from rl_bgd.replay.memory import estimate_sequence_replay_memory
from rl_bgd.replay.sequence_buffer import SequenceReplayBuffer
from rl_bgd.utils.checkpoint_io import save_training_checkpoint as _save_training_checkpoint
from rl_bgd.utils.checkpoint_payload import (
    checkpoint_finite_float,
    checkpoint_float_list,
    checkpoint_float_mapping,
    checkpoint_generator_state,
    checkpoint_observation,
)
from rl_bgd.utils.checkpoint_progress import (
    checkpoint_boolean,
    checkpoint_integer,
    checkpoint_step,
)
from rl_bgd.utils.checkpoint_transaction import transactional_state_load
from rl_bgd.utils.config_validation import (
    config_nonnegative_integer,
    config_positive_integer,
)
from rl_bgd.utils.randomness import (
    load_random_state_dict,
    preserved_random_state,
    random_state_dict,
)

PostStepObserver = Callable[[int, RecurrentSACAgent], None]

_RECURRENT_SAC_TRAINING_CHECKPOINT_VERSION = 2


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
    replay_storage: Literal["model", "cpu", "auto"] = "model"

    def validate(self) -> None:
        config_positive_integer(self.total_steps, name="total_steps")
        config_nonnegative_integer(self.random_steps, name="random_steps")
        config_positive_integer(self.sequence_batch_size, name="sequence_batch_size")
        config_nonnegative_integer(self.burn_in, name="burn_in")
        config_positive_integer(self.unroll, name="unroll")
        config_positive_integer(self.replay_capacity, name="replay_capacity")
        config_positive_integer(self.updates_per_step, name="updates_per_step")
        config_nonnegative_integer(self.seed, name="seed")
        if self.replay_storage not in ("model", "cpu", "auto"):
            raise ValueError("replay_storage must be model, cpu or auto")
        if self.replay_capacity < self.burn_in + self.unroll:
            raise ValueError("replay capacity is shorter than one sequence window")


def _replay_storage_device(
    agent_device: torch.device,
    config: RecurrentSACTrainConfig,
    *,
    observation_dim: int,
    action_dim: int,
) -> torch.device:
    """Select storage only; never alter model device, sampling or replay capacity."""
    if config.replay_storage == "model":
        return agent_device
    if config.replay_storage == "cpu" or agent_device.type != "cuda":
        return torch.device("cpu")
    estimate = estimate_sequence_replay_memory(
        config.replay_capacity,
        observation_dim,
        action_dim,
        batch_size=config.sequence_batch_size,
        burn_in=config.burn_in,
        unroll=config.unroll,
    )
    free_bytes, _ = torch.cuda.mem_get_info(device=agent_device)
    required_bytes = (
        estimate.total_storage_bytes
        + estimate.sampled_batch_bytes
        + estimate.sampling_index_bytes_upper_bound
    )
    # Reserve at least half the currently available device RAM for the
    # agent, gradients, optimizer, simulator and transient update buffers.
    return agent_device if required_bytes <= 0.5 * free_bytes else torch.device("cpu")


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
    if checkpoint_interval is not None:
        config_positive_integer(checkpoint_interval, name="checkpoint_interval")
    if checkpoint_interval is not None and checkpoint_path is None:
        raise ValueError("checkpoint_interval requires checkpoint_path")
    if max_steps_this_call is not None:
        config_positive_integer(max_steps_this_call, name="max_steps_this_call")

    observation_dim = int(env.observation_space.low.numel())
    action_dim = int(env.action_space.low.numel())
    storage_device = _replay_storage_device(
        agent.device,
        config,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    replay = SequenceReplayBuffer(
        config.replay_capacity,
        observation_dim,
        action_dim,
        storage_device=storage_device,
    )
    generator = torch.Generator(device=agent.device).manual_seed(config.seed + 27)
    # The original model-resident sampler remains bit-for-bit unchanged.
    # A separate sampler is required only if replay lives on another device.
    sampling_generator = (
        generator
        if storage_device == agent.device
        else torch.Generator(device=storage_device).manual_seed(config.seed + 271)
    )
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
            device=storage_device if storage_device.type == "cpu" else agent.device,
        )
        checkpoint_version = checkpoint_integer(
            checkpoint.get("version"),
            name="recurrent SAC training checkpoint version",
        )
        if checkpoint_version not in (1, _RECURRENT_SAC_TRAINING_CHECKPOINT_VERSION):
            raise ValueError("unsupported recurrent SAC training checkpoint version")
        expected_config = asdict(config)
        if checkpoint_version == 1:
            if config.replay_storage != "model":
                raise ValueError("legacy recurrent SAC checkpoints require model replay storage")
            expected_config.pop("replay_storage")
        if checkpoint.get("train_config") != expected_config:
            raise ValueError("recurrent SAC training checkpoint configuration mismatch")
        if checkpoint_version == 2 and checkpoint.get("replay_storage_device") != str(
            storage_device
        ):
            raise ValueError("recurrent SAC replay storage device mismatch")
        start_step = checkpoint_step(
            checkpoint.get("next_step"),
            name="recurrent SAC checkpoint next_step",
            limit=config.total_steps,
        )
        episode_start = checkpoint_boolean(
            checkpoint.get("episode_start"),
            name="recurrent SAC checkpoint episode_start",
        )

        agent_state = checkpoint.get("agent")
        replay_state = checkpoint.get("replay")
        environment_state = checkpoint.get("environment")
        if not isinstance(agent_state, dict) or not isinstance(replay_state, dict):
            raise TypeError("recurrent SAC agent/replay state must be dictionaries")
        if not isinstance(environment_state, dict):
            raise TypeError("recurrent SAC environment state must be a dictionary")
        replay_transition_id = checkpoint_integer(
            replay_state.get("next_transition_id"),
            name="recurrent SAC replay checkpoint next_transition_id",
        )
        if replay_transition_id != start_step:
            raise ValueError("recurrent SAC checkpoint replay/step progress mismatch")

        observation = checkpoint_observation(
            checkpoint.get("observation"),
            name="recurrent SAC checkpoint observation",
            device=agent.device,
            expected_shape=env.observation_space.shape,
        )
        raw_history = checkpoint.get("episode_history")
        if not isinstance(raw_history, list):
            raise TypeError("recurrent SAC episode history must be a list")
        episode_history = [
            checkpoint_observation(
                item,
                name=f"recurrent SAC episode_history[{index}]",
                device=agent.device,
                expected_shape=env.observation_space.shape,
            )
            for index, item in enumerate(raw_history)
        ]
        episode_return = checkpoint_finite_float(
            checkpoint.get("episode_return"),
            name="recurrent SAC checkpoint episode_return",
        )
        completed_returns = checkpoint_float_list(
            checkpoint.get("completed_returns"),
            name="recurrent SAC completed_returns",
        )
        last_metrics = checkpoint_float_mapping(
            checkpoint.get("last_metrics"),
            name="recurrent SAC last_metrics",
        )
        generator_state = checkpoint_generator_state(
            checkpoint.get("replay_generator_state"),
            name="recurrent SAC replay generator checkpoint",
            device=agent.device,
        )
        sampler_generator_state: Tensor | None = None
        if sampling_generator is not generator:
            if checkpoint_version != 2:
                raise ValueError("separate replay sampler requires a version 2 checkpoint")
            sampler_generator_state = checkpoint_generator_state(
                checkpoint.get("sampler_generator_state"),
                name="recurrent SAC storage sampler checkpoint",
                device=storage_device,
            )
        elif checkpoint_version == 2 and checkpoint.get("sampler_generator_state") is not None:
            raise ValueError("unexpected separate replay sampler in checkpoint")
        process_rng = checkpoint.get("process_rng")
        if not isinstance(process_rng, dict):
            raise TypeError("recurrent SAC process RNG state must be a dictionary")
        with preserved_random_state():
            load_random_state_dict(process_rng)

        resume_state = {
            "agent": agent_state,
            "replay": replay_state,
            "environment": environment_state,
            "replay_generator_state": generator_state,
            "sampler_generator_state": sampler_generator_state,
            "process_rng": process_rng,
        }

        def current_resume_state() -> dict[str, Any]:
            return {
                "agent": agent.state_dict(),
                "replay": replay.state_dict(),
                "environment": _environment_state_dict(env),
                "replay_generator_state": generator.get_state().clone(),
                "sampler_generator_state": (
                    sampling_generator.get_state().clone()
                    if sampling_generator is not generator else None
                ),
                "process_rng": random_state_dict(),
            }

        def apply_resume(payload: dict[str, Any]) -> None:
            agent.load_state_dict(payload["agent"])
            replay.load_state_dict(payload["replay"])
            _load_environment_state(env, payload["environment"])
            generator.set_state(payload["replay_generator_state"].cpu())
            if sampling_generator is not generator:
                sampling_generator.set_state(payload["sampler_generator_state"].cpu())
            load_random_state_dict(payload["process_rng"])

        transactional_state_load(
            resume_state,
            current_state=current_resume_state,
            apply=apply_resume,
        )

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
                "episode_history": [item.detach().clone() for item in episode_history],
                "episode_return": episode_return,
                "completed_returns": list(completed_returns),
                "last_metrics": dict(last_metrics),
                "replay_generator_state": generator.get_state().clone(),
                "sampler_generator_state": (
                    sampling_generator.get_state().clone()
                    if sampling_generator is not generator else None
                ),
                "replay_storage_device": str(storage_device),
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
                    generator=sampling_generator,
                )
                last_metrics = agent.update(batch.to(agent.device))
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
        "replay_storage_device": str(storage_device),
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
