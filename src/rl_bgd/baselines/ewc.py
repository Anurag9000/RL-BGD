"""Elastic Weight Consolidation regularizers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.baselines.importance import (
    ParameterState,
    quadratic_importance_penalty,
    snapshot_parameters,
    validate_importance,
)
from rl_bgd.utils.checkpoint_progress import checkpoint_finite_float, checkpoint_integer


@dataclass
class EWCConsolidation:
    """One anchor and diagonal importance estimate."""

    anchor: ParameterState
    importance: ParameterState


def _clone_state(
    state: Mapping[str, Tensor],
) -> ParameterState:
    return {name: value.detach().float().clone() for name, value in state.items()}


class EWCRegularizer:
    """Multi-consolidation diagonal EWC.

    Consolidation timing is intentionally external. Supplying true task
    boundaries makes this a boundary-aware/oracle baseline, not a strict
    task-agnostic method.
    """

    def __init__(
        self,
        strength: float = 1.0,
    ) -> None:
        if strength < 0:
            raise ValueError("EWC strength must be non-negative")
        self.strength = float(strength)
        self.states: list[EWCConsolidation] = []

    def consolidate(
        self,
        module: nn.Module,
        importance: Mapping[
            str,
            Tensor,
        ],
    ) -> None:
        validate_importance(
            module,
            importance,
        )
        self.states.append(
            EWCConsolidation(
                anchor=snapshot_parameters(module),
                importance=_clone_state(importance),
            )
        )

    def penalty(
        self,
        module: nn.Module,
    ) -> Tensor:
        if not self.states:
            parameter = next(module.parameters())
            return torch.zeros(
                (),
                device=parameter.device,
                dtype=torch.float32,
            )
        total = torch.zeros(
            (),
            device=next(module.parameters()).device,
            dtype=torch.float32,
        )
        for state in self.states:
            total = total + (
                quadratic_importance_penalty(
                    module,
                    state.anchor,
                    state.importance,
                )
            )
        return 0.5 * self.strength * total

    def state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "strength": self.strength,
            "states": [
                {
                    "anchor": _clone_state(state.anchor),
                    "importance": _clone_state(state.importance),
                }
                for state in self.states
            ],
        }

    def load_state_dict(
        self,
        state: Mapping[
            str,
            Any,
        ],
    ) -> None:
        version = checkpoint_integer(
            state.get("version"),
            name="EWC checkpoint version",
        )
        if version != 1:
            raise ValueError("unsupported EWC checkpoint version")
        strength = checkpoint_finite_float(
            state.get("strength"),
            name="EWC checkpoint strength",
        )
        if strength < 0:
            raise ValueError("invalid EWC strength in checkpoint")
        incoming = state.get("states")
        if not isinstance(incoming, list):
            raise TypeError("EWC checkpoint states must be a list")
        staged_states: list[EWCConsolidation] = []
        for item in incoming:
            if not isinstance(item, Mapping):
                raise TypeError("EWC checkpoint consolidation must be a mapping")
            anchor = item.get("anchor")
            importance = item.get("importance")
            if not isinstance(anchor, Mapping) or not isinstance(importance, Mapping):
                raise TypeError("EWC checkpoint anchor and importance must be mappings")
            staged_states.append(
                EWCConsolidation(
                    anchor=_clone_state(anchor),
                    importance=_clone_state(importance),
                )
            )
        self.strength = strength
        self.states = staged_states


class OnlineEWCRegularizer:
    """Single-anchor online EWC with exponentially decayed importance."""

    def __init__(
        self,
        strength: float = 1.0,
        *,
        decay: float = 1.0,
    ) -> None:
        if strength < 0:
            raise ValueError("Online-EWC strength must be non-negative")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("Online-EWC decay must lie in [0, 1]")
        self.strength = float(strength)
        self.decay = float(decay)
        self.anchor: ParameterState | None = None
        self.importance: ParameterState | None = None

    def consolidate(
        self,
        module: nn.Module,
        importance: Mapping[
            str,
            Tensor,
        ],
    ) -> None:
        validate_importance(
            module,
            importance,
        )
        incoming = _clone_state(importance)
        if self.importance is None:
            merged = incoming
        else:
            if set(self.importance) != set(incoming):
                raise ValueError("Online-EWC importance keys changed")
            merged = {
                name: (self.decay * self.importance[name] + incoming[name]) for name in incoming
            }
        self.importance = merged
        self.anchor = snapshot_parameters(module)

    def penalty(
        self,
        module: nn.Module,
    ) -> Tensor:
        if self.anchor is None or self.importance is None:
            parameter = next(module.parameters())
            return torch.zeros(
                (),
                device=parameter.device,
                dtype=torch.float32,
            )
        return (
            0.5
            * self.strength
            * quadratic_importance_penalty(
                module,
                self.anchor,
                self.importance,
            )
        )

    def state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "strength": self.strength,
            "decay": self.decay,
            "anchor": (None if self.anchor is None else _clone_state(self.anchor)),
            "importance": (None if self.importance is None else _clone_state(self.importance)),
        }

    def load_state_dict(
        self,
        state: Mapping[
            str,
            Any,
        ],
    ) -> None:
        version = checkpoint_integer(
            state.get("version"),
            name="Online-EWC checkpoint version",
        )
        if version != 1:
            raise ValueError("unsupported Online-EWC checkpoint version")
        strength = checkpoint_finite_float(
            state.get("strength"),
            name="Online-EWC checkpoint strength",
        )
        decay = checkpoint_finite_float(
            state.get("decay"),
            name="Online-EWC checkpoint decay",
        )
        if strength < 0 or not 0.0 <= decay <= 1.0:
            raise ValueError("invalid Online-EWC checkpoint hyperparameters")
        anchor = state.get("anchor")
        importance = state.get("importance")
        if (anchor is None) != (importance is None):
            raise ValueError("Online-EWC checkpoint has incomplete consolidated state")
        if anchor is not None and not isinstance(anchor, Mapping):
            raise TypeError("Online-EWC checkpoint anchor must be a mapping")
        if importance is not None and not isinstance(importance, Mapping):
            raise TypeError("Online-EWC checkpoint importance must be a mapping")
        staged_anchor = None if anchor is None else _clone_state(anchor)
        staged_importance = None if importance is None else _clone_state(importance)
        self.strength = strength
        self.decay = decay
        self.anchor = staged_anchor
        self.importance = staged_importance
