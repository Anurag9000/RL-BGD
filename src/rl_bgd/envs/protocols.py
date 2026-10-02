"""Shared structural protocols for tensor-native continuous environments."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class TensorSpace(Protocol):
    """Read-only tensor-space surface used by collection loops."""

    @property
    def low(self) -> Tensor: ...

    @property
    def high(self) -> Tensor: ...

    def sample(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


class ContinuousTensorEnv(Protocol):
    """Minimal tensor-native Gymnasium-style environment contract."""

    @property
    def action_space(self) -> TensorSpace: ...

    @property
    def observation_space(self) -> TensorSpace: ...

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[
        Tensor,
        dict[str, object],
    ]: ...

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


class TaskAwareTensorEnv(
    ContinuousTensorEnv,
    Protocol,
):
    """Continuous environment whose current task index is trainer-visible."""

    @property
    def cur_seq_idx(self) -> int: ...
