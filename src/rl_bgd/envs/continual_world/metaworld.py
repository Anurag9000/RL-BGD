"""Modern Meta-World 3.x adapter for the Continual World protocol."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_bgd.envs.continual_world.stream import (
    ContinualWorldBenchmark,
    ContinualWorldStreamConfig,
    ContinualWorldStreamEnv,
    continual_world_task_sequence,
)
from rl_bgd.envs.synthetic.lqr import TensorBox


class MetaWorldTaskAdapter:
    """Convert a Meta-World task into the tensor-native environment contract."""

    def __init__(
        self,
        env: Any,
        task: Any,
        *,
        device: torch.device | str = "cpu",
        horizon: int = 200,
    ) -> None:
        if horizon < 1:
            raise ValueError(
                "Meta-World horizon must be positive"
            )
        self.env = env
        self.device = torch.device(device)
        self.horizon = horizon
        self._episode_step = 0

        self.env.set_task(task)
        # Continual World's random_init_all mode fixes the task definition
        # but deliberately re-samples the environment reset vector.
        if hasattr(
            self.env,
            "_freeze_rand_vec",
        ):
            self.env._freeze_rand_vec = False

        self.action_space = TensorBox(
            low=torch.as_tensor(
                self.env.action_space.low,
                device=self.device,
                dtype=torch.float32,
            ).clone(),
            high=torch.as_tensor(
                self.env.action_space.high,
                device=self.device,
                dtype=torch.float32,
            ).clone(),
        )
        self.observation_space = TensorBox(
            low=torch.as_tensor(
                self.env.observation_space.low,
                device=self.device,
                dtype=torch.float32,
            ).clone(),
            high=torch.as_tensor(
                self.env.observation_space.high,
                device=self.device,
                dtype=torch.float32,
            ).clone(),
        )

    def _observation_tensor(
        self,
        observation: Any,
    ) -> Tensor:
        return torch.as_tensor(
            observation,
            device=self.device,
            dtype=torch.float32,
        ).clone()

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[
        Tensor,
        dict[str, object],
    ]:
        reset_output = self.env.reset(
            seed=seed
        )
        if (
            isinstance(
                reset_output,
                tuple,
            )
            and len(reset_output) == 2
        ):
            observation, _ = reset_output
        else:
            observation = reset_output
        self._episode_step = 0
        return (
            self._observation_tensor(
                observation
            ),
            {},
        )

    def step(
        self,
        action: Tensor,
    ) -> tuple[
        Tensor,
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        # MuJoCo/Meta-World is CPU-hosted, so this transfer is an explicit
        # simulator boundary rather than an accidental training fallback.
        cpu_action = (
            action.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .numpy()
        )
        step_output = self.env.step(
            cpu_action
        )
        if len(step_output) == 5:
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = step_output
        elif len(step_output) == 4:
            (
                observation,
                reward,
                done,
                info,
            ) = step_output
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError(
                "unexpected Meta-World step return signature"
            )

        self._episode_step += 1
        if (
            self._episode_step
            >= self.horizon
            and not terminated
        ):
            truncated = True

        safe_info: dict[str, object] = {}
        if (
            isinstance(info, dict)
            and "success" in info
        ):
            safe_info["success"] = float(
                info["success"]
            )
        return (
            self._observation_tensor(
                observation
            ),
            float(reward),
            bool(terminated),
            bool(truncated),
            safe_info,
        )


def make_continual_world_stream(
    benchmark: ContinualWorldBenchmark,
    *,
    steps_per_task: int = 1_000_000,
    seed: int = 0,
    device: torch.device | str = "cpu",
    episode_horizon: int = 200,
) -> ContinualWorldStreamEnv:
    """Build strict CW10/CW20 on current Meta-World without legacy task IDs."""

    try:
        import metaworld
    except ImportError as exc:
        raise RuntimeError(
            "Meta-World is required for Continual World; "
            "install the 'continual-world' extra"
        ) from exc

    task_names = list(
        continual_world_task_sequence(
            benchmark
        )
    )
    benchmark_api = metaworld.MT50(
        seed=seed
    )
    train_classes = (
        benchmark_api.train_classes
    )
    train_tasks = (
        benchmark_api.train_tasks
    )

    envs: list[MetaWorldTaskAdapter] = []
    for occurrence, task_name in enumerate(
        task_names
    ):
        if task_name not in train_classes:
            raise RuntimeError(
                "current Meta-World does not provide "
                f"required Continual World task {task_name}"
            )
        matching_tasks = [
            task
            for task in train_tasks
            if task.env_name == task_name
        ]
        if not matching_tasks:
            raise RuntimeError(
                "Meta-World task bank is missing "
                f"{task_name}"
            )
        env = train_classes[
            task_name
        ]()
        # A fixed task definition with _freeze_rand_vec=False matches the
        # published random_init_all protocol while modern Meta-World handles
        # reset randomization itself.
        adapter = MetaWorldTaskAdapter(
            env,
            matching_tasks[0],
            device=device,
            horizon=episode_horizon,
        )
        # Distinct instances are intentional for the repeated CW20 sequence.
        del occurrence
        envs.append(adapter)

    return ContinualWorldStreamEnv(
        envs,
        task_names,
        config=ContinualWorldStreamConfig(
            steps_per_task=steps_per_task,
            strict_task_agnostic=True,
        ),
    )
