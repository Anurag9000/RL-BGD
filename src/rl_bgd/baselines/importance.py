"""Shared diagonal-importance utilities for continual-learning baselines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import torch
from torch import Tensor, nn

ParameterState = dict[str, Tensor]
LossClosure = Callable[[], Tensor]
OutputClosure = Callable[[], Tensor]


def trainable_parameters(
    module: nn.Module,
) -> dict[str, nn.Parameter]:
    parameters = {
        name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    if not parameters:
        raise ValueError("module has no trainable parameters")
    return parameters


def snapshot_parameters(
    module: nn.Module,
) -> ParameterState:
    """Clone trainable parameters without preserving autograd history."""

    return {
        name: parameter.detach().float().clone()
        for name, parameter in trainable_parameters(module).items()
    }


def zeros_like_parameters(
    module: nn.Module,
) -> ParameterState:
    return {
        name: torch.zeros_like(
            parameter.detach(),
            dtype=torch.float32,
        )
        for name, parameter in trainable_parameters(module).items()
    }


def validate_importance(
    module: nn.Module,
    importance: Mapping[str, Tensor],
) -> None:
    parameters = trainable_parameters(module)
    if set(importance) != set(parameters):
        raise ValueError("importance keys do not match trainable parameters")
    for name, parameter in parameters.items():
        value = importance[name]
        if value.shape != parameter.shape:
            raise ValueError(f"importance shape mismatch for {name}")
        if not torch.isfinite(value).all() or torch.any(value < 0):
            raise ValueError(f"importance must be finite and non-negative for {name}")


def quadratic_importance_penalty(
    module: nn.Module,
    anchor: Mapping[str, Tensor],
    importance: Mapping[str, Tensor],
) -> Tensor:
    """Return sum_i importance_i * (theta_i-anchor_i)^2."""

    parameters = trainable_parameters(module)
    if set(anchor) != set(parameters) or set(importance) != set(parameters):
        raise ValueError("anchor/importance keys do not match trainable parameters")
    penalty = torch.zeros(
        (),
        device=next(iter(parameters.values())).device,
        dtype=torch.float32,
    )
    for name, parameter in parameters.items():
        anchor_value = anchor[name].to(
            parameter.device,
            dtype=torch.float32,
        )
        importance_value = importance[name].to(
            parameter.device,
            dtype=torch.float32,
        )
        if anchor_value.shape != parameter.shape or importance_value.shape != parameter.shape:
            raise ValueError(f"regularizer shape mismatch for {name}")
        penalty = penalty + (importance_value * (parameter.float() - anchor_value).square()).sum()
    return penalty


def empirical_fisher_diagonal(
    module: nn.Module,
    loss_closures: Iterable[LossClosure],
) -> ParameterState:
    """Estimate diagonal empirical Fisher as mean squared sample gradients.

    Each closure should recompute one scalar negative-log-likelihood-like loss
    from the current module parameters. RL callers may instead use a documented
    surrogate score; that choice must be reported because it changes the
    baseline definition.
    """

    parameters = trainable_parameters(module)
    names = list(parameters)
    parameter_tuple = tuple(parameters[name] for name in names)
    total = {
        name: torch.zeros_like(
            parameter.detach(),
            dtype=torch.float32,
        )
        for name, parameter in parameters.items()
    }
    count = 0
    for closure in loss_closures:
        loss = closure()
        if loss.ndim != 0:
            raise ValueError("Fisher loss closure must return a scalar")
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite Fisher loss")
        gradients = torch.autograd.grad(
            loss,
            parameter_tuple,
            allow_unused=False,
            create_graph=False,
        )
        for (
            name,
            gradient,
        ) in zip(
            names,
            gradients,
            strict=True,
        ):
            total[name].add_(gradient.detach().float().square())
        count += 1
    if count == 0:
        raise ValueError("at least one Fisher loss closure is required")
    return {name: value / float(count) for name, value in total.items()}


def mas_importance(
    module: nn.Module,
    output_closures: Iterable[OutputClosure],
) -> ParameterState:
    """Estimate MAS importance from output-function sensitivity.

    For each sample this differentiates one-half the squared L2 norm of the
    model output and averages the absolute parameter gradients.
    """

    parameters = trainable_parameters(module)
    names = list(parameters)
    parameter_tuple = tuple(parameters[name] for name in names)
    total = {
        name: torch.zeros_like(
            parameter.detach(),
            dtype=torch.float32,
        )
        for name, parameter in parameters.items()
    }
    count = 0
    for closure in output_closures:
        output = closure()
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite MAS model output")
        objective = 0.5 * output.float().square().sum()
        gradients = torch.autograd.grad(
            objective,
            parameter_tuple,
            allow_unused=False,
            create_graph=False,
        )
        for (
            name,
            gradient,
        ) in zip(
            names,
            gradients,
            strict=True,
        ):
            total[name].add_(gradient.detach().float().abs())
        count += 1
    if count == 0:
        raise ValueError("at least one MAS output closure is required")
    return {name: value / float(count) for name, value in total.items()}
