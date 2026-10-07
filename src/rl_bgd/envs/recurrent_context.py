"""Task-agnostic recurrent context observation adapters."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_bgd.envs.protocols import ContinuousEnv
from rl_bgd.envs.synthetic.lqr import TensorBox


class PreviousTransitionContextEnv:
    """Append previous action, reward, and done to each observation.

    The wrapper does not expose task identity or task-boundary metadata. At an
    episode reset, previous action/reward are zero and previous_done is one.
    For a transition (s_t, a_t, r_t, s_{t+1}), the returned next observation is
    [s_{t+1}, a_t, r_t, done_t].
    """

    def __init__(
        self,
        env: ContinuousEnv,
        *,
        reward_bound: float = 1_000_000.0,
    ) -> None:
        if reward_bound <= 0:
            raise ValueError("reward_bound must be positive")
        self.env = env
        self.reward_bound = float(reward_bound)
        self.action_space = env.action_space
        observation_low = env.observation_space.low
        observation_high = env.observation_space.high
        action_low = env.action_space.low.to(
            device=observation_low.device,
            dtype=observation_low.dtype,
        )
        action_high = env.action_space.high.to(
            device=observation_high.device,
            dtype=observation_high.dtype,
        )
        reward_low = torch.tensor(
            [-reward_bound],
            device=observation_low.device,
            dtype=observation_low.dtype,
        )
        reward_high = torch.tensor(
            [reward_bound],
            device=observation_high.device,
            dtype=observation_high.dtype,
        )
        done_low = torch.zeros(
            1,
            device=observation_low.device,
            dtype=observation_low.dtype,
        )
        done_high = torch.ones(
            1,
            device=observation_high.device,
            dtype=observation_high.dtype,
        )
        self.observation_space = TensorBox(
            low=torch.cat(
                [
                    observation_low,
                    action_low,
                    reward_low,
                    done_low,
                ]
            ),
            high=torch.cat(
                [
                    observation_high,
                    action_high,
                    reward_high,
                    done_high,
                ]
            ),
        )

    @property
    def raw_observation_dim(self) -> int:
        return int(self.env.observation_space.low.numel())

    @property
    def context_layout(self) -> dict[str, int]:
        return {
            "observation": self.raw_observation_dim,
            "previous_action": int(self.action_space.low.numel()),
            "previous_reward": 1,
            "previous_done": 1,
        }

    def _augment(
        self,
        observation: Tensor,
        *,
        previous_action: Tensor,
        previous_reward: float | Tensor,
        previous_done: bool,
    ) -> Tensor:
        obs = observation.to(
            device=self.observation_space.low.device,
            dtype=self.observation_space.low.dtype,
        ).reshape(-1)
        action = previous_action.to(
            device=obs.device,
            dtype=obs.dtype,
        ).reshape(-1)
        reward = torch.as_tensor(
            previous_reward,
            device=obs.device,
            dtype=obs.dtype,
        ).reshape(1)
        done = torch.tensor(
            [float(previous_done)],
            device=obs.device,
            dtype=obs.dtype,
        )
        return torch.cat([obs, action, reward, done])

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        observation, info = self.env.reset(seed=seed)
        return (
            self._augment(
                observation,
                previous_action=torch.zeros_like(self.action_space.low),
                previous_reward=0.0,
                previous_done=True,
            ),
            dict(info),
        )

    def step(
        self,
        action: Tensor,
    ) -> tuple[Tensor, float, bool, bool, dict[str, object]]:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)
        return (
            self._augment(
                observation,
                previous_action=action,
                previous_reward=reward,
                previous_done=terminated or truncated,
            ),
            reward,
            terminated,
            truncated,
            dict(info),
        )

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def state_dict(self) -> dict[str, Any]:
        state_fn = getattr(self.env, "state_dict", None)
        if not callable(state_fn):
            raise TypeError("wrapped environment does not support checkpointing")
        return {
            "version": 2,
            "reward_bound": self.reward_bound,
            "env": state_fn(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 2:
            raise ValueError("unsupported context-wrapper checkpoint version")
        if state.get("reward_bound") != self.reward_bound:
            raise ValueError("context-wrapper checkpoint reward bound mismatch")
        load_fn = getattr(self.env, "load_state_dict", None)
        if not callable(load_fn):
            raise TypeError("wrapped environment does not support checkpoint restore")
        nested = state.get("env")
        if not isinstance(nested, dict):
            raise TypeError("context-wrapper nested checkpoint must be a mapping")
        load_fn(nested)
