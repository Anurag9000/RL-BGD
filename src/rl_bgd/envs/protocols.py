"""Shared structural protocols for tensor-native continuous-control environments."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class TensorContinuousSpace(Protocol):
    """Minimal tensor-space contract required by RL collection loops."""

    @property
    def low(self) -> Tensor: ...

    @property
    def high(self) -> Tensor: ...

    def sample(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


class ContinuousEnv(Protocol):
    """Tensor-native continuous environment used by SAC/PPO trainers."""

    @property
    def action_space(self) -> TensorContinuousSpace: ...

    @property
    def observation_space(self) -> TensorContinuousSpace: ...

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, object]]: ...

    def step(
        self,
        action: Tensor,
    ) -> tuple[
        Tensor,
        float,
        bool,
        bool,
        dict[str, object],
    ]: ...


class TaskAwareContinuousEnv(ContinuousEnv, Protocol):
    """Continuous environment that exposes the active task sequence index."""

    @property
    def cur_seq_idx(self) -> int: ...


# Explicit aliases used by trainer signatures. They name the same structural
# tensor-native contracts rather than introducing duplicate protocols.
ContinuousTensorEnv = ContinuousEnv
TaskAwareTensorEnv = TaskAwareContinuousEnv
