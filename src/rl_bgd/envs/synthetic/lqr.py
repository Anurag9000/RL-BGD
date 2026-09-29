"""Small continuous-control environment for dependency-light RL validation."""

from __future__ import annotations

from dataclasses import dataclass

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
        return self.low + (
            self.high - self.low
        ) * torch.rand(
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
            raise ValueError(
                "dimension and horizon must be positive"
            )
        if action_cost <= 0 or process_noise < 0:
            raise ValueError(
                "invalid cost/noise parameters"
            )
        self.dimension = dimension
        self.horizon = horizon
        self.dynamics = dynamics
        self.control_gain = control_gain
        self.action_cost = action_cost
        self.process_noise = process_noise
        self.device = torch.device(device)
        low = torch.full(
            (dimension,), -1.0, device=self.device
        )
        high = torch.full(
            (dimension,), 1.0, device=self.device
        )
        self.action_space = TensorBox(
            low=low, high=high
        )
        self.observation_space = TensorBox(
            low=torch.full(
                (dimension,), -10.0, device=self.device
            ),
            high=torch.full(
                (dimension,), 10.0, device=self.device
            ),
        )
        self._state = torch.zeros(
            dimension, device=self.device
        )
        self._step = 0
        self._generator = torch.Generator(
            device=self.device
        )

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[Tensor, dict[str, object]]:
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
        control_cost = (
            self.action_cost * action.square().sum()
        )
        reward = -(state_cost + control_cost)
        noise = torch.zeros_like(self._state)
        if self.process_noise:
            noise = self.process_noise * torch.randn(
                self._state.shape,
                device=self.device,
                generator=self._generator,
            )
        self._state = (
            self.dynamics * self._state
            + self.control_gain * action
            + noise
        )
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
