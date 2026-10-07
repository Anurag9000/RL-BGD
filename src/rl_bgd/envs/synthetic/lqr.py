"""Small continuous-control environment for dependency-light RL validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class TensorBox:
    low: Tensor
    high: Tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.low.shape)

    def sample(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.low + (self.high - self.low) * torch.rand(
            self.low.shape,
            device=self.low.device,
            generator=generator,
        )


class LinearQuadraticControlEnv:
    """Finite-horizon linear dynamics with quadratic state/action cost."""

    def __init__(
        self,
        *,
        dimension: int = 1,
        horizon: int = 50,
        dynamics: float = 0.9,
        control_gain: float = 0.5,
        action_cost: float = 0.05,
        process_noise: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if dimension < 1 or horizon < 1:
            raise ValueError("dimension and horizon must be positive")
        if action_cost <= 0 or process_noise < 0:
            raise ValueError("invalid cost/noise parameters")
        self.dimension = dimension
        self.horizon = horizon
        self.dynamics = dynamics
        self.control_gain = control_gain
        self.action_cost = action_cost
        self.process_noise = process_noise
        self.device = torch.device(device)
        self._initial_config = {
            "dimension": dimension,
            "horizon": horizon,
            "dynamics": float(dynamics),
            "control_gain": float(control_gain),
            "action_cost": float(action_cost),
            "process_noise": float(process_noise),
        }
        low = torch.full((dimension,), -1.0, device=self.device)
        high = torch.full((dimension,), 1.0, device=self.device)
        self.action_space = TensorBox(low=low, high=high)
        self.observation_space = TensorBox(
            low=torch.full((dimension,), -10.0, device=self.device),
            high=torch.full((dimension,), 10.0, device=self.device),
        )
        self._state = torch.zeros(dimension, device=self.device)
        self._step = 0
        self._generator = torch.Generator(device=self.device)

    def reset(self, *, seed: int | None = None) -> tuple[Tensor, dict[str, object]]:
        if seed is not None:
            self._generator.manual_seed(seed)
        self._state = (
            2.0
            * torch.rand(
                (self.dimension,),
                device=self.device,
                generator=self._generator,
            )
            - 1.0
        )
        self._step = 0
        return self._state.clone(), {}

    def step(
        self, action: Tensor
    ) -> tuple[
        Tensor,
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        action = action.to(self.device).clamp(
            self.action_space.low,
            self.action_space.high,
        )
        state_cost = self._state.square().sum()
        control_cost = self.action_cost * action.square().sum()
        reward = -(state_cost + control_cost)
        noise = torch.zeros_like(self._state)
        if self.process_noise:
            noise = self.process_noise * torch.randn(
                self._state.shape,
                device=self.device,
                generator=self._generator,
            )
        self._state = self.dynamics * self._state + self.control_gain * action + noise
        self._step += 1
        terminated = False
        truncated = self._step >= self.horizon
        return (
            self._state.clone(),
            float(reward.item()),
            terminated,
            truncated,
            {},
        )


    def state_dict(self) -> dict[str, Any]:
        """Serialize simulator, local RNG, and mutable context parameters."""

        return {
            "version": 1,
            "initial_config": dict(self._initial_config),
            "current_parameters": {
                "dynamics": float(self.dynamics),
                "control_gain": float(self.control_gain),
                "action_cost": float(self.action_cost),
                "process_noise": float(self.process_noise),
            },
            "state": self._state.detach().clone(),
            "step": self._step,
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported LQR checkpoint version")
        if state.get("initial_config") != self._initial_config:
            raise ValueError("LQR checkpoint configuration mismatch")

        saved_state = state.get("state")
        generator_state = state.get("generator_state")
        step = state.get("step")
        current = state.get("current_parameters")
        if not isinstance(saved_state, Tensor) or saved_state.shape != self._state.shape:
            raise ValueError("LQR checkpoint state shape mismatch")
        if not torch.isfinite(saved_state).all():
            raise ValueError("LQR checkpoint contains nonfinite simulator state")
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= self.horizon:
            raise ValueError("LQR checkpoint step is invalid")
        if not isinstance(generator_state, Tensor):
            raise TypeError("LQR checkpoint generator state must be a tensor")
        if not isinstance(current, dict):
            raise TypeError("LQR checkpoint current parameters must be a mapping")

        required = {
            "dynamics",
            "control_gain",
            "action_cost",
            "process_noise",
        }
        if set(current) != required:
            raise ValueError("LQR checkpoint current parameters are incomplete")
        values = {name: float(current[name]) for name in required}
        if values["action_cost"] <= 0 or values["process_noise"] < 0:
            raise ValueError("LQR checkpoint contains invalid mutable parameters")

        self.dynamics = values["dynamics"]
        self.control_gain = values["control_gain"]
        self.action_cost = values["action_cost"]
        self.process_noise = values["process_noise"]
        self._state.copy_(
            saved_state.to(
                device=self.device,
                dtype=self._state.dtype,
            )
        )
        self._step = step
        self._generator.set_state(generator_state.cpu())
