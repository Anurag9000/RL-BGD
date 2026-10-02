"""Live smoke for CORA's historical CHORES/ALFRED runtime."""

from __future__ import annotations

import json
import os
from typing import Any

import gym
import numpy as np
from continual_rl.experiments.tasks.make_chores_task import get_chores_task
from continual_rl.utils.utils import Utils

_DEMO = (
    "pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-402/"
    "trial_T20210817_071626_357261"
)


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
        raise RuntimeError("CORA CHORES environment step must return a tuple")
    if len(output) == 4:
        observation, reward, done, info = output
    elif len(output) == 5:
        observation, reward, terminated, truncated, info = output
        done = bool(terminated) or bool(truncated)
    else:
        raise RuntimeError(
            f"unexpected CORA CHORES step signature length: {len(output)}"
        )
    if not isinstance(info, dict):
        raise TypeError("CORA CHORES step info must be a dictionary")
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise FloatingPointError("CORA CHORES returned a nonfinite reward")
    return observation, reward_value, bool(done), info


def main() -> None:
    data_dir = os.environ.get("ALFRED_DATA_DIR")
    if not data_dir:
        raise RuntimeError("ALFRED_DATA_DIR must point to the CORA CHORES data")
    expected_demo = os.path.join(
        data_dir,
        "train",
        _DEMO,
        "traj_data.json",
    )
    if not os.path.isfile(expected_demo):
        raise FileNotFoundError(
            "official CORA CHORES trajectory was not found: "
            f"{expected_demo}"
        )

    task = get_chores_task(
        "rl_bgd_cora_chores_smoke",
        "train",
        [_DEMO],
        num_timesteps=16,
        eval_mode=False,
        max_episode_steps=8,
        continual_eval=False,
    )
    task_spec = task._task_spec  # CORA exposes no public TaskSpec accessor.
    env, _ = Utils.make_env(
        task_spec.env_spec,
        seed_to_set=37,
    )
    try:
        observation = _reset(env)
        array = np.asarray(observation)
        if array.size == 0 or not np.isfinite(array.astype(np.float32)).all():
            raise RuntimeError("CORA CHORES reset returned an invalid observation")

        action = env.action_space.sample()
        next_observation, reward, done, info = _step(
            env,
            action,
        )
        next_array = np.asarray(next_observation)
        if (
            next_array.size == 0
            or not np.isfinite(next_array.astype(np.float32)).all()
        ):
            raise RuntimeError("CORA CHORES step returned an invalid observation")

        import ai2thor
        import crl_alfred

        print(
            json.dumps(
                {
                    "cora_task_id": task_spec.task_id,
                    "environment": "CHORES-ALFRED",
                    "demo": _DEMO,
                    "gym_version": gym.__version__,
                    "ai2thor_version": getattr(
                        ai2thor,
                        "__version__",
                        "unknown",
                    ),
                    "crl_alfred_version": getattr(
                        crl_alfred,
                        "__version__",
                        "0.1",
                    ),
                    "observation_shape": list(array.shape),
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
