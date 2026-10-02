"""Live smoke for CORA's isolated historical MiniHack runtime."""

from __future__ import annotations

import json
from typing import Any

from continual_rl.experiments.tasks.make_minihack_task import (
    get_single_minihack_task,
)
from continual_rl.utils.utils import Utils
import gym
import minihack
import nle
import numpy as np


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
        raise RuntimeError("CORA MiniHack environment step must return a tuple")
    if len(output) == 4:
        observation, reward, done, info = output
    elif len(output) == 5:
        observation, reward, terminated, truncated, info = output
        done = bool(terminated) or bool(truncated)
    else:
        raise RuntimeError(
            f"unexpected CORA MiniHack step signature length: {len(output)}"
        )
    if not isinstance(info, dict):
        raise TypeError("CORA MiniHack step info must be a dictionary")
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise FloatingPointError("CORA MiniHack returned a nonfinite reward")
    return observation, reward_value, bool(done), info


def main() -> None:
    task = get_single_minihack_task(
        "rl_bgd_cora_minihack_smoke",
        0,
        "Room-Random-5x5-v0",
        num_timesteps=16,
        eval_mode=False,
    )
    task_spec = task._task_spec  # CORA exposes no public TaskSpec accessor.
    env, _ = Utils.make_env(
        task_spec.env_spec,
        seed_to_set=31,
    )
    try:
        observation = _reset(env)
        if observation is None:
            raise RuntimeError("CORA MiniHack reset returned no observation")
        action = env.action_space.sample()
        next_observation, reward, done, info = _step(env, action)
        if next_observation is None:
            raise RuntimeError("CORA MiniHack step returned no observation")
        print(
            json.dumps(
                {
                    "cora_task_id": task_spec.task_id,
                    "environment": "MiniHack-Room-Random-5x5-v0",
                    "gym_version": gym.__version__,
                    "minihack_version": getattr(minihack, "__version__", "unknown"),
                    "nle_version": getattr(nle, "__version__", "unknown"),
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
