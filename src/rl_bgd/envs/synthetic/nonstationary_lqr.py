"""Boundary-free nonstationary wrapper for the synthetic LQR environment."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from torch import Tensor

from rl_bgd.continual.schedules import ContextSchedule
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv, TensorBox
from rl_bgd.utils.checkpoint_progress import checkpoint_integer, checkpoint_nonnegative_integer

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

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "schedule_config": asdict(self.schedule.config),
            "environment_step": self.environment_step,
            "current_context": dict(self._current_context),
            "base_env": self.base_env.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = checkpoint_integer(
            state.get("version"),
            name="scheduled-LQR checkpoint version",
        )
        if version != 1:
            raise ValueError("unsupported scheduled-LQR checkpoint version")
        if state.get("schedule_config") != asdict(self.schedule.config):
            raise ValueError("scheduled-LQR checkpoint schedule mismatch")
        environment_step = checkpoint_nonnegative_integer(
            state.get("environment_step"),
            name="scheduled-LQR environment_step",
        )
        current_context = state.get("current_context")
        base_state = state.get("base_env")
        if not isinstance(current_context, dict):
            raise TypeError("scheduled-LQR current context must be a mapping")
        if not isinstance(base_state, dict):
            raise TypeError("scheduled-LQR base environment state must be a mapping")

        normalized: dict[str, float] = {}
        for key, value in current_context.items():
            if not isinstance(key, str):
                raise TypeError("scheduled-LQR context keys must be strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"scheduled-LQR context {key} must be numeric")
            normalized[key] = float(value)
        expected_keys = set(self.schedule.config.anchors[0])
        if set(normalized) != expected_keys:
            raise ValueError("scheduled-LQR checkpoint context keys do not match the schedule")
        if not all(math.isfinite(value) for value in normalized.values()):
            raise ValueError("scheduled-LQR checkpoint context contains nonfinite values")

        saved_parameters = base_state.get("current_parameters")
        if not isinstance(saved_parameters, dict):
            raise TypeError("scheduled-LQR base current parameters must be a mapping")
        for name, value in normalized.items():
            base_value = saved_parameters.get(name)
            if isinstance(base_value, bool) or not isinstance(base_value, (int, float)):
                raise TypeError(f"scheduled-LQR base parameter {name} must be numeric")
            if not math.isfinite(float(base_value)) or float(base_value) != value:
                raise ValueError("scheduled-LQR checkpoint context disagrees with base environment")

        base_step = checkpoint_integer(
            base_state.get("step"),
            name="scheduled-LQR base environment step",
        )
        if base_step < 0 or base_step > self.base_env.horizon:
            raise ValueError("scheduled-LQR base environment step is invalid")
        if base_step > environment_step:
            raise ValueError("scheduled-LQR checkpoint environment progress is inconsistent")
        context_step = environment_step if base_step == 0 else environment_step - 1
        if context_step < 0:
            raise ValueError("scheduled-LQR checkpoint environment progress is inconsistent")
        expected_context = {
            key: float(value) for key, value in self.schedule.context_at(context_step).items()
        }
        if normalized != expected_context:
            raise ValueError("scheduled-LQR checkpoint context does not match the schedule")

        self.base_env.load_state_dict(base_state)
        self.environment_step = environment_step
        self._current_context = normalized
