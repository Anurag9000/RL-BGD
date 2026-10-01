"""Strict task-agnostic adapter for CARL 1.1.x environments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rl_bgd.continual.schedules import ContextSchedule
from rl_bgd.envs.synthetic.lqr import TensorBox


class CARLImportError(ImportError):
    """Raised when the optional CARL dependency is unavailable or incompatible."""


class CARLContextStream:
    """Expose a CARL environment as a tensor-native hidden-context stream.

    CARL observations contain both obs and context. In strict mode this adapter
    returns only obs and removes context_id from info. Context changes are
    generated solely from the environment-step schedule and are never reported
    to the agent as task IDs or boundary callbacks.

    CARL selects contexts on reset. Exact per-step trajectories therefore apply
    the scheduled context through CARLEnv's context state and environment-
    specific context update hook.
    """

    def __init__(
        self,
        env: Any,
        schedule: ContextSchedule,
        *,
        device: torch.device | str = "cpu",
        strict_task_agnostic: bool = True,
        expose_evaluation_context: bool = False,
    ) -> None:
        if strict_task_agnostic and expose_evaluation_context:
            raise ValueError("strict task-agnostic mode cannot expose evaluation context")
        self.env = env
        self.schedule = schedule
        self.device = torch.device(device)
        self.strict_task_agnostic = strict_task_agnostic
        self.expose_evaluation_context = expose_evaluation_context
        self.environment_step = 0
        self._current_context: dict[str, float] = {}
        self.action_space = self._tensor_box(env.action_space)
        base_space = getattr(env, "base_observation_space", None)
        if base_space is None:
            observation_space = getattr(env, "observation_space", None)
            if observation_space is None:
                raise TypeError("CARL environment has no observation space")
            spaces = getattr(observation_space, "spaces", {})
            base_space = spaces.get("obs")
        if base_space is None:
            raise TypeError("cannot identify CARL base observation space")
        self.observation_space = self._tensor_box(base_space)

    def _tensor_box(self, space: Any) -> TensorBox:
        if not hasattr(space, "low") or not hasattr(space, "high"):
            raise TypeError("SAC CARL adapter requires continuous Box spaces")
        low = torch.as_tensor(
            np.asarray(space.low),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        high = torch.as_tensor(
            np.asarray(space.high),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        return TensorBox(low=low, high=high)

    def _apply_context(self) -> None:
        context = self.schedule.context_at(self.environment_step)
        context_space = self.env.get_context_space()
        context_with_defaults = context_space.insert_defaults(context)
        self.env.context = context_with_defaults
        update_context = getattr(self.env, "_update_context", None)
        if update_context is None:
            raise TypeError("CARL environment lacks context update hook")
        update_context()
        self._current_context = dict(context)

    def _observation(self, value: Any) -> Tensor:
        if isinstance(value, dict):
            if "obs" not in value:
                raise KeyError("CARL observation dictionary lacks 'obs'")
            value = value["obs"]
        tensor = torch.as_tensor(
            np.asarray(value),
            dtype=torch.float32,
            device=self.device,
        )
        return tensor.reshape(-1)

    def _info(self, info: dict[str, Any]) -> dict[str, Any]:
        result = dict(info)
        if self.strict_task_agnostic:
            result.pop("context_id", None)
            result.pop("context", None)
        if self.expose_evaluation_context:
            result["evaluation_context"] = dict(self._current_context)
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        self._apply_context()
        return self._observation(observation), self._info(info)

    def step(
        self,
        action: Tensor,
    ) -> tuple[Tensor, float, bool, bool, dict[str, Any]]:
        self._apply_context()
        numpy_action = action.detach().to("cpu").numpy().reshape(self.env.action_space.shape)
        observation, reward, terminated, truncated, info = self.env.step(numpy_action)
        self.environment_step += 1
        return (
            self._observation(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(info),
        )

    @property
    def evaluation_context(self) -> dict[str, float]:
        """Return context for evaluator-owned diagnostics, never policy input."""

        return dict(self._current_context)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


def make_carl_pendulum_stream(
    schedule: ContextSchedule,
    *,
    device: torch.device | str = "cpu",
    strict_task_agnostic: bool = True,
    expose_evaluation_context: bool = False,
) -> CARLContextStream:
    """Create a CARLPendulum stream without importing CARL at package import."""

    try:
        envs = import_module("carl.envs")
        selectors = import_module("carl.context.selection")
    except ImportError as exc:
        raise CARLImportError(
            "CARL is optional. Install the pinned CARL environment extra in "
            "a compatible environment: pip install -e '.[carl]'."
        ) from exc

    carl_pendulum = getattr(envs, "CARLPendulum", None)
    static_selector = getattr(selectors, "StaticSelector", None)
    if carl_pendulum is None or static_selector is None:
        raise CARLImportError("installed CARL package does not expose the 1.1.x API")

    initial = schedule.context_at(0)
    env = carl_pendulum(
        contexts={0: initial},
        context_selector=static_selector,
        # CARL v1.1.1 contains a Pendulum context-name mismatch between
        # releases. Strict task-agnostic runs do not consume context
        # observations anyway, so request no context features at the source.
        obs_context_features=([] if strict_task_agnostic else list(initial)),
        obs_context_as_dict=True,
    )
    return CARLContextStream(
        env,
        schedule,
        device=device,
        strict_task_agnostic=strict_task_agnostic,
        expose_evaluation_context=expose_evaluation_context,
    )
