"""Live smoke for CORA's isolated legacy Procgen runtime."""

from __future__ import annotations

import json
from typing import Any

import gym
import numpy as np
import procgen

from continual_rl.experiments.tasks.make_procgen_task import (
    get_single_procgen_task,
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
        raise RuntimeError("CORA Procgen environment step must return a tuple")
    if len(output) == 4:
        observation, reward, done, info = output
    elif len(output) == 5:
        observation, reward, terminated, truncated, info = output
        done = bool(terminated) or bool(truncated)
    else:
        raise RuntimeError(f"unexpected CORA Procgen step signature length: {len(output)}")
    if not isinstance(info, dict):
        raise TypeError("CORA Procgen step info must be a dictionary")
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise FloatingPointError("CORA Procgen returned a nonfinite reward")
    return observation, reward_value, bool(done), info


def main() -> None:
    task = get_single_procgen_task(
        "rl_bgd_cora_climber_smoke",
        0,
        "climber-v0",
        num_timesteps=16,
        eval_mode=False,
        num_levels=1,
        start_level=0,
        distribution_mode="easy",
    )
    task_spec = task._task_spec  # CORA exposes no public TaskSpec accessor.
    env, _ = Utils.make_env(
        task_spec.env_spec,
        seed_to_set=23,
    )
    try:
        observation = _reset(env)
        if observation is None:
            raise RuntimeError("CORA Procgen reset returned no observation")
        action = env.action_space.sample()
        next_observation, reward, done, info = _step(
            env,
            action,
        )
        if next_observation is None:
            raise RuntimeError("CORA Procgen step returned no observation")

        print(
            json.dumps(
                {
                    "cora_task_id": task_spec.task_id,
                    "environment": "procgen-climber-v0",
                    "gym_version": gym.__version__,
                    "procgen_version": getattr(procgen, "__version__", "unknown"),
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
