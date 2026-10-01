"""Modern Meta-World 3.x adapters for Continual World protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from rl_bgd.envs.continual_world.canonical import (
    CanonicalContinualWorldConfig,
    CanonicalContinualWorldStreamEnv,
    TaskIdentityObservationEnv,
)
from rl_bgd.envs.continual_world.stream import (
    ContinualWorldBenchmark,
    ContinualWorldStreamConfig,
    ContinualWorldStreamEnv,
    ContinuousTaskEnv,
    continual_world_task_sequence,
)
from rl_bgd.envs.synthetic.lqr import TensorBox

ContinualWorldProtocol = Literal["canonical", "task_agnostic"]


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
            raise ValueError("Meta-World horizon must be positive")
        self.env = env
        self.device = torch.device(device)
        self.horizon = horizon
        self._episode_step = 0
        self._closed = False

        self.env.set_task(task)
        # Continual World's random_init_all mode fixes the task definition
        # but deliberately re-samples the environment reset vector.
        if hasattr(self.env, "_freeze_rand_vec"):
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

    def _observation_tensor(self, observation: Any) -> Tensor:
        return torch.as_tensor(
            observation,
            device=self.device,
            dtype=torch.float32,
        ).clone()

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        reset_output = self.env.reset(seed=seed)
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            observation, _ = reset_output
        else:
            observation = reset_output
        self._episode_step = 0
        return self._observation_tensor(observation), {}

    def step(
        self,
        action: Tensor,
    ) -> tuple[Tensor, float, bool, bool, dict[str, object]]:
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
        step_output = self.env.step(cpu_action)
        if len(step_output) == 5:
            observation, reward, terminated, truncated, info = step_output
        elif len(step_output) == 4:
            observation, reward, done, info = step_output
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError("unexpected Meta-World step return signature")

        self._episode_step += 1
        if self._episode_step >= self.horizon and not terminated:
            truncated = True

        safe_info: dict[str, object] = {}
        if isinstance(info, dict) and "success" in info:
            safe_info["success"] = float(info["success"])
        return (
            self._observation_tensor(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            safe_info,
        )

    def close(self) -> None:
        """Close the underlying simulator exactly once."""

        if self._closed:
            return
        close = getattr(self.env, "close", None)
        if close is not None:
            close()
        self._closed = True


def _build_metaworld_task_adapters(
    benchmark: ContinualWorldBenchmark,
    *,
    seed: int,
    device: torch.device | str,
    episode_horizon: int,
) -> tuple[list[MetaWorldTaskAdapter], list[str]]:
    try:
        import metaworld
    except ImportError as exc:
        raise RuntimeError(
            "Meta-World is required for Continual World; install the 'continual-world' extra"
        ) from exc

    task_names = list(continual_world_task_sequence(benchmark))
    benchmark_api = metaworld.MT50(seed=seed)
    train_classes = benchmark_api.train_classes
    train_tasks = benchmark_api.train_tasks

    envs: list[MetaWorldTaskAdapter] = []
    for task_name in task_names:
        if task_name not in train_classes:
            raise RuntimeError(
                f"current Meta-World does not provide required Continual World task {task_name}"
            )
        matching_tasks = [task for task in train_tasks if task.env_name == task_name]
        if not matching_tasks:
            raise RuntimeError(f"Meta-World task bank is missing {task_name}")
        env = train_classes[task_name]()
        envs.append(
            MetaWorldTaskAdapter(
                env,
                matching_tasks[0],
                device=device,
                horizon=episode_horizon,
            )
        )
    return envs, task_names


def make_continual_world_stream(
    benchmark: ContinualWorldBenchmark,
    *,
    steps_per_task: int = 1_000_000,
    seed: int = 0,
    device: torch.device | str = "cpu",
    episode_horizon: int = 200,
) -> ContinualWorldStreamEnv:
    """Build strict TA-CW10/TA-CW20 without task IDs or switch callbacks."""

    envs, task_names = _build_metaworld_task_adapters(
        benchmark,
        seed=seed,
        device=device,
        episode_horizon=episode_horizon,
    )
    return ContinualWorldStreamEnv(
        envs,
        task_names,
        config=ContinualWorldStreamConfig(
            steps_per_task=steps_per_task,
            strict_task_agnostic=True,
        ),
    )


def make_canonical_continual_world_stream(
    benchmark: ContinualWorldBenchmark,
    *,
    steps_per_task: int = 1_000_000,
    seed: int = 0,
    device: torch.device | str = "cpu",
    episode_horizon: int = 200,
) -> CanonicalContinualWorldStreamEnv:
    """Build the task-aware CW10/CW20 observation/boundary protocol."""

    base_envs, task_names = _build_metaworld_task_adapters(
        benchmark,
        seed=seed,
        device=device,
        episode_horizon=episode_horizon,
    )
    num_tasks = len(base_envs)
    envs: list[ContinuousTaskEnv] = [
        TaskIdentityObservationEnv(
            env,
            task_index=index,
            num_tasks=num_tasks,
        )
        for index, env in enumerate(base_envs)
    ]
    return CanonicalContinualWorldStreamEnv(
        envs,
        task_names,
        config=CanonicalContinualWorldConfig(
            steps_per_task=steps_per_task,
        ),
    )


def make_continual_world_evaluation_envs(
    benchmark: ContinualWorldBenchmark,
    *,
    protocol: ContinualWorldProtocol,
    seed: int = 0,
    device: torch.device | str = "cpu",
    episode_horizon: int = 200,
) -> tuple[ContinuousTaskEnv, ...]:
    """Create evaluation-only task instances matched to the chosen protocol."""

    base_envs, _ = _build_metaworld_task_adapters(
        benchmark,
        seed=seed,
        device=device,
        episode_horizon=episode_horizon,
    )
    if protocol == "task_agnostic":
        return tuple(base_envs)
    if protocol == "canonical":
        num_tasks = len(base_envs)
        return tuple(
            TaskIdentityObservationEnv(
                env,
                task_index=index,
                num_tasks=num_tasks,
            )
            for index, env in enumerate(base_envs)
        )
    raise ValueError(f"unsupported Continual World protocol: {protocol}")


@dataclass(frozen=True)
class ContinualWorldProtocolBundle:
    """Keep training and evaluation environments physically separate."""

    train_env: ContinualWorldStreamEnv | CanonicalContinualWorldStreamEnv
    evaluation_envs: tuple[ContinuousTaskEnv, ...]
    task_names: tuple[str, ...]
    protocol: ContinualWorldProtocol

    def close(self) -> None:
        """Close training and evaluation environments owned by the bundle."""

        self.train_env.close()
        for env in self.evaluation_envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()


def make_continual_world_protocol(
    benchmark: ContinualWorldBenchmark,
    *,
    protocol: ContinualWorldProtocol,
    steps_per_task: int = 1_000_000,
    seed: int = 0,
    device: torch.device | str = "cpu",
    episode_horizon: int = 200,
) -> ContinualWorldProtocolBundle:
    """Construct matched train/evaluation environments for CW experiments."""

    if protocol == "task_agnostic":
        train_env: ContinualWorldStreamEnv | CanonicalContinualWorldStreamEnv = (
            make_continual_world_stream(
                benchmark,
                steps_per_task=steps_per_task,
                seed=seed,
                device=device,
                episode_horizon=episode_horizon,
            )
        )
    elif protocol == "canonical":
        train_env = make_canonical_continual_world_stream(
            benchmark,
            steps_per_task=steps_per_task,
            seed=seed,
            device=device,
            episode_horizon=episode_horizon,
        )
    else:
        raise ValueError(f"unsupported Continual World protocol: {protocol}")

    evaluation_envs = make_continual_world_evaluation_envs(
        benchmark,
        protocol=protocol,
        seed=seed + 100_000,
        device=device,
        episode_horizon=episode_horizon,
    )
    return ContinualWorldProtocolBundle(
        train_env=train_env,
        evaluation_envs=evaluation_envs,
        task_names=continual_world_task_sequence(benchmark),
        protocol=protocol,
    )
