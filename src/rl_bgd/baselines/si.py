"""Synaptic Intelligence continual-learning regularizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.baselines.importance import (
    ParameterState,
    quadratic_importance_penalty,
    snapshot_parameters,
    trainable_parameters,
    zeros_like_parameters,
)


class SynapticIntelligence:
    """Track optimization-path contribution and consolidate parameter importance.

    Call capture_gradients before the optimizer step and accumulate_after_step
    immediately after it. Consolidation timing remains explicit because a true
    task-boundary trigger would not be available in strict task-agnostic runs.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        strength: float = 1.0,
        damping: float = 0.1,
    ) -> None:
        if strength < 0:
            raise ValueError(
                "SI strength must be non-negative"
            )
        if damping <= 0:
            raise ValueError(
                "SI damping must be positive"
            )
        self.strength = float(
            strength
        )
        self.damping = float(
            damping
        )
        self.anchor = snapshot_parameters(
            module
        )
        self.previous = snapshot_parameters(
            module
        )
        self.path_integral = zeros_like_parameters(
            module
        )
        self.importance = zeros_like_parameters(
            module
        )

    def capture_gradients(
        self,
        module: nn.Module,
    ) -> ParameterState:
        parameters = trainable_parameters(
            module
        )
        gradients: ParameterState = {}
        for name, parameter in parameters.items():
            if parameter.grad is None:
                raise RuntimeError(
                    f"SI requires a gradient for {name}"
                )
            if not torch.isfinite(
                parameter.grad
            ).all():
                raise FloatingPointError(
                    f"nonfinite SI gradient for {name}"
                )
            gradients[
                name
            ] = (
                parameter.grad.detach()
                .float()
                .clone()
            )
        return gradients

    def accumulate_after_step(
        self,
        module: nn.Module,
        gradients: Mapping[
            str,
            Tensor,
        ],
    ) -> None:
        parameters = trainable_parameters(
            module
        )
        if (
            set(
                gradients
            )
            != set(
                parameters
            )
        ):
            raise ValueError(
                "SI gradient keys do not match trainable parameters"
            )
        for name, parameter in parameters.items():
            current = (
                parameter.detach().float()
            )
            previous = self.previous[
                name
            ].to(
                current.device
            )
            gradient = gradients[
                name
            ].to(
                current.device,
                dtype=torch.float32,
            )
            if (
                gradient.shape
                != current.shape
            ):
                raise ValueError(
                    f"SI gradient shape mismatch for {name}"
                )
            delta = (
                current - previous
            )
            self.path_integral[
                name
            ] = (
                self.path_integral[
                    name
                ].to(
                    current.device
                )
                - gradient
                * delta
            )
            self.previous[
                name
            ] = current.clone()

    def consolidate(
        self,
        module: nn.Module,
    ) -> None:
        current = snapshot_parameters(
            module
        )
        if (
            set(
                current
            )
            != set(
                self.anchor
            )
        ):
            raise ValueError(
                "SI parameter set changed before consolidation"
            )
        for name, value in current.items():
            anchor = self.anchor[
                name
            ].to(
                value.device
            )
            displacement = (
                value - anchor
            )
            contribution = torch.clamp(
                self.path_integral[
                    name
                ].to(
                    value.device
                ),
                min=0.0,
            ) / (
                displacement.square()
                + self.damping
            )
            self.importance[
                name
            ] = (
                self.importance[
                    name
                ].to(
                    value.device
                )
                + contribution
            )
            self.anchor[
                name
            ] = value.clone()
            self.previous[
                name
            ] = value.clone()
            self.path_integral[
                name
            ] = torch.zeros_like(
                value
            )

    def penalty(
        self,
        module: nn.Module,
    ) -> Tensor:
        return (
            self.strength
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
            state: Mapping[
                str,
                Tensor,
            ],
        ) -> ParameterState:
            return {
                name: value.detach().float().clone()
                for name, value in state.items()
            }

        return {
            "version": 1,
            "strength": self.strength,
            "damping": self.damping,
            "anchor": clone(
                self.anchor
            ),
            "previous": clone(
                self.previous
            ),
            "path_integral": clone(
                self.path_integral
            ),
            "importance": clone(
                self.importance
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[
            str,
            Any,
        ],
    ) -> None:
        if state.get(
            "version"
        ) != 1:
            raise ValueError(
                "unsupported SI checkpoint version"
            )
        self.strength = float(
            state[
                "strength"
            ]
        )
        self.damping = float(
            state[
                "damping"
            ]
        )
        if (
            self.strength < 0
            or self.damping <= 0
        ):
            raise ValueError(
                "invalid SI checkpoint hyperparameters"
            )

        def clone_field(
            name: str,
        ) -> ParameterState:
            field = state[
                name
            ]
            if not isinstance(
                field,
                Mapping,
            ):
                raise TypeError(
                    f"SI checkpoint {name} must be a mapping"
                )
            return {
                key: value.detach().float().clone()
                for key, value in field.items()
            }

        self.anchor = clone_field(
            "anchor"
        )
        self.previous = clone_field(
            "previous"
        )
        self.path_integral = clone_field(
            "path_integral"
        )
        self.importance = clone_field(
            "importance"
        )
