"""Boundary-free nonstationary wrapper for the synthetic LQR environment."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from rl_bgd.continual.schedules import ContextSchedule
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv, TensorBox

_ALLOWED_CONTEXT_KEYS = {
    "dynamics",
    "control_gain",
    "action_cost",
    "process_noise",
}


class ScheduledLQREnv:
    """Apply a context schedule without exposing context or switch events."""

    def __init__(
        self,
        base_env: LinearQuadraticControlEnv,
        schedule: ContextSchedule,
    ) -> None:
        keys = set(schedule.config.anchors[0])
        unsupported = keys - _ALLOWED_CONTEXT_KEYS
        if unsupported:
            raise ValueError(f"unsupported LQR context keys: {sorted(unsupported)}")
        self.base_env = base_env
        self.schedule = schedule
        self.environment_step = 0
        self.action_space: TensorBox = base_env.action_space
        self.observation_space: TensorBox = base_env.observation_space
        self._current_context: dict[str, float] = {}
        self._apply_context()

    def _apply_context(self) -> None:
        context = self.schedule.context_at(self.environment_step)
        if "action_cost" in context and context["action_cost"] <= 0:
            raise ValueError("scheduled action_cost must be positive")
        if "process_noise" in context and context["process_noise"] < 0:
            raise ValueError("scheduled process_noise cannot be negative")
        for name, value in context.items():
            setattr(self.base_env, name, float(value))
        self._current_context = dict(context)

    @property
    def evaluation_context(self) -> dict[str, float]:
        """Evaluator-only current context; not returned by reset or step."""

        return dict(self._current_context)

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        self._apply_context()
        observation, _ = self.base_env.reset(seed=seed)
        return observation, {}

    def step(
        self,
        action: Tensor,
    ) -> tuple[Tensor, float, bool, bool, dict[str, Any]]:
        self._apply_context()
        observation, reward, terminated, truncated, _ = self.base_env.step(action)
        self.environment_step += 1
        return observation, reward, terminated, truncated, {}
