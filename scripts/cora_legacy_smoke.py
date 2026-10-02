"""Live smoke for the isolated legacy CORA Atari runtime.

This script is intentionally dependency-agnostic with respect to RL-BGD.  The
corresponding workflow installs the pinned upstream CORA revision and its
historical Gym/ALE-compatible runtime in a separate Python environment.
"""

from __future__ import annotations

import json
from typing import Any

import gym
import numpy as np
from continual_rl.experiments.tasks.make_atari_task import (
    get_single_atari_task,
)
from continual_rl.utils.utils import Utils


def _reset(env: Any) -> Any:
    output = env.reset()
    if isinstance(output, tuple):
        return output[0]
    return output


def _step(
    env: Any,
    action: Any,
) -> tuple[Any, float, bool, dict[str, Any]]:
    output = env.step(action)
    if not isinstance(output, tuple):
        raise RuntimeError("CORA Atari environment step must return a tuple")
    if len(output) == 4:
        observation, reward, done, info = output
    elif len(output) == 5:
        observation, reward, terminated, truncated, info = output
        done = bool(terminated) or bool(truncated)
    else:
        raise RuntimeError(
            f"unexpected CORA Atari step signature length: {len(output)}"
        )
    if not isinstance(info, dict):
        raise TypeError("CORA Atari step info must be a dictionary")
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise FloatingPointError("CORA Atari returned a nonfinite reward")
    return observation, reward_value, bool(done), info


def main() -> None:
    task = get_single_atari_task(
        "rl_bgd_cora_pong_smoke",
        0,
        "PongNoFrameskip-v4",
        num_timesteps=16,
        max_episode_steps=100,
        full_action_space=True,
    )
    task_spec = task._task_spec  # CORA exposes no public TaskSpec accessor.
    env, _ = Utils.make_env(
        task_spec.env_spec,
        seed_to_set=17,
    )
    try:
        observation = _reset(env)
        if observation is None:
            raise RuntimeError("CORA Atari reset returned no observation")
        action = env.action_space.sample()
        next_observation, reward, done, info = _step(
            env,
            action,
        )
        if next_observation is None:
            raise RuntimeError("CORA Atari step returned no observation")

        print(
            json.dumps(
                {
                    "cora_task_id": task_spec.task_id,
                    "environment": "PongNoFrameskip-v4",
                    "gym_version": gym.__version__,
                    "action_space": str(env.action_space),
                    "reward": reward,
                    "done": done,
                    "info_keys": sorted(str(key) for key in info),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
