"""Memory Aware Synapses regularizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.baselines.importance import (
    ParameterState,
    checkpoint_parameter_state,
    quadratic_importance_penalty,
    snapshot_parameters,
    validate_checkpoint_parameter_layout,
    validate_importance,
)
from rl_bgd.utils.checkpoint_progress import checkpoint_finite_float, checkpoint_integer


class MASRegularizer:
    """Multi-consolidation MAS quadratic penalty."""

    def __init__(
        self,
        strength: float = 1.0,
    ) -> None:
        if strength < 0:
            raise ValueError("MAS strength must be non-negative")
        self.strength = float(strength)
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
        incoming = {name: value.detach().float().clone() for name, value in importance.items()}
        if self.importance is None:
            self.importance = incoming
        else:
            if set(self.importance) != set(incoming):
                raise ValueError("MAS parameter set changed")
            self.importance = {
                name: (self.importance[name].to(incoming[name].device) + incoming[name])
                for name in incoming
            }
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
        def clone(
            state: ParameterState | None,
        ) -> ParameterState | None:
            if state is None:
                return None
            return {name: value.detach().float().clone() for name, value in state.items()}

        return {
            "version": 1,
            "strength": self.strength,
            "anchor": clone(self.anchor),
            "importance": clone(self.importance),
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
            name="MAS checkpoint version",
        )
        if version != 1:
            raise ValueError("unsupported MAS checkpoint version")
        strength = checkpoint_finite_float(
            state.get("strength"),
            name="MAS checkpoint strength",
        )
        if strength < 0:
            raise ValueError("invalid MAS checkpoint strength")
        anchor = state.get("anchor")
        importance = state.get("importance")
        if (anchor is None) != (importance is None):
            raise ValueError("MAS checkpoint has incomplete state")

        staged_anchor = (
            None
            if anchor is None
            else checkpoint_parameter_state(
                anchor,
                name="MAS checkpoint anchor",
            )
        )
        staged_importance = (
            None
            if importance is None
            else checkpoint_parameter_state(
                importance,
                name="MAS checkpoint importance",
                nonnegative=True,
            )
        )
        if staged_anchor is not None and staged_importance is not None:
            validate_checkpoint_parameter_layout(
                staged_anchor,
                staged_importance,
                name="MAS checkpoint importance",
            )
        self.strength = strength
        self.anchor = staged_anchor
        self.importance = staged_importance
