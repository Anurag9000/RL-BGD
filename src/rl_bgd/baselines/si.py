"""Synaptic Intelligence continual-learning regularizer."""

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
    trainable_parameters,
    validate_checkpoint_parameter_layout,
    zeros_like_parameters,
)
from rl_bgd.utils.checkpoint_progress import checkpoint_finite_float, checkpoint_integer


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
            raise ValueError("SI strength must be non-negative")
        if damping <= 0:
            raise ValueError("SI damping must be positive")
        self.strength = float(strength)
        self.damping = float(damping)
        self.anchor = snapshot_parameters(module)
        self.previous = snapshot_parameters(module)
        self.path_integral = zeros_like_parameters(module)
        self.importance = zeros_like_parameters(module)

    def capture_gradients(
        self,
        module: nn.Module,
    ) -> ParameterState:
        parameters = trainable_parameters(module)
        gradients: ParameterState = {}
        for name, parameter in parameters.items():
            if parameter.grad is None:
                raise RuntimeError(f"SI requires a gradient for {name}")
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"nonfinite SI gradient for {name}")
            gradients[name] = parameter.grad.detach().float().clone()
        return gradients

    def accumulate_after_step(
        self,
        module: nn.Module,
        gradients: Mapping[
            str,
            Tensor,
        ],
    ) -> None:
        parameters = trainable_parameters(module)
        if set(gradients) != set(parameters):
            raise ValueError("SI gradient keys do not match trainable parameters")
        if set(self.previous) != set(parameters) or set(self.path_integral) != set(parameters):
            raise ValueError("SI accumulated parameter keys do not match the model")

        staged_previous: ParameterState = {}
        staged_path_integral: ParameterState = {}
        for name, parameter in parameters.items():
            current = parameter.detach().float()
            previous = self.previous[name].to(current.device)
            path = self.path_integral[name].to(current.device)
            gradient = gradients[name].to(
                current.device,
                dtype=torch.float32,
            )
            if (
                current.shape != previous.shape
                or current.shape != path.shape
                or current.shape != gradient.shape
            ):
                raise ValueError(f"SI accumulation shape mismatch for {name}")
            if not all(
                torch.isfinite(tensor).all().item()
                for tensor in (current, previous, path, gradient)
            ):
                raise FloatingPointError(f"nonfinite SI accumulation input for {name}")
            updated_path = path - gradient * (current - previous)
            if not torch.isfinite(updated_path).all().item():
                raise FloatingPointError(f"nonfinite SI accumulated path for {name}")
            staged_path_integral[name] = updated_path
            staged_previous[name] = current.clone()

        self.path_integral = staged_path_integral
        self.previous = staged_previous

    def consolidate(
        self,
        module: nn.Module,
    ) -> None:
        current = snapshot_parameters(module)
        for state in (self.anchor, self.previous, self.path_integral, self.importance):
            validate_checkpoint_parameter_layout(
                current,
                state,
                name="SI consolidation",
            )

        staged_importance: ParameterState = {}
        for name, value in current.items():
            anchor = self.anchor[name].to(value.device)
            path = self.path_integral[name].to(value.device)
            importance = self.importance[name].to(value.device)
            if not all(
                torch.isfinite(tensor).all().item() for tensor in (value, anchor, path, importance)
            ):
                raise FloatingPointError(f"nonfinite SI consolidation input for {name}")
            if torch.any(importance < 0).item():
                raise ValueError(f"SI importance must be non-negative for {name}")
            displacement = value - anchor
            contribution = torch.clamp(path, min=0.0) / (displacement.square() + self.damping)
            updated_importance = importance + contribution
            if not torch.isfinite(updated_importance).all().item():
                raise FloatingPointError(f"nonfinite SI consolidated importance for {name}")
            staged_importance[name] = updated_importance

        self.importance = staged_importance
        self.anchor = {name: value.clone() for name, value in current.items()}
        self.previous = {name: value.clone() for name, value in current.items()}
        self.path_integral = {name: torch.zeros_like(value) for name, value in current.items()}

    def penalty(
        self,
        module: nn.Module,
    ) -> Tensor:
        return self.strength * quadratic_importance_penalty(
            module,
            self.anchor,
            self.importance,
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
            return {name: value.detach().float().clone() for name, value in state.items()}

        return {
            "version": 1,
            "strength": self.strength,
            "damping": self.damping,
            "anchor": clone(self.anchor),
            "previous": clone(self.previous),
            "path_integral": clone(self.path_integral),
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
            name="SI checkpoint version",
        )
        if version != 1:
            raise ValueError("unsupported SI checkpoint version")
        strength = checkpoint_finite_float(
            state.get("strength"),
            name="SI checkpoint strength",
        )
        damping = checkpoint_finite_float(
            state.get("damping"),
            name="SI checkpoint damping",
        )
        if strength < 0 or damping <= 0:
            raise ValueError("invalid SI checkpoint hyperparameters")

        anchor = checkpoint_parameter_state(
            state.get("anchor"),
            name="SI checkpoint anchor",
        )
        previous = checkpoint_parameter_state(
            state.get("previous"),
            name="SI checkpoint previous",
        )
        path_integral = checkpoint_parameter_state(
            state.get("path_integral"),
            name="SI checkpoint path_integral",
        )
        importance = checkpoint_parameter_state(
            state.get("importance"),
            name="SI checkpoint importance",
            nonnegative=True,
        )
        for field_name, candidate in (
            ("previous", previous),
            ("path_integral", path_integral),
            ("importance", importance),
        ):
            validate_checkpoint_parameter_layout(
                anchor,
                candidate,
                name=f"SI checkpoint {field_name}",
            )
        validate_checkpoint_parameter_layout(
            self.anchor,
            anchor,
            name="SI checkpoint live model",
        )
        self.strength = strength
        self.damping = damping
        self.anchor = anchor
        self.previous = previous
        self.path_integral = path_integral
        self.importance = importance
