"""Uncertainty-guided continual learning regularization primitives.

The implementation follows the PPO-UCL regularizer in csm9493/UCL while
repairing the checked-in RL code's inconsistent bias-posterior interface.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class UCLBayesianLayer(Protocol):
    in_features: int
    weight_mu: Tensor
    weight_rho: Tensor
    bias_mu: Tensor
    bias_rho: Tensor


@dataclass(frozen=True)
class UCLLayerSnapshot:
    weight_mu: Tensor
    weight_sigma: Tensor
    bias_mu: Tensor
    bias_sigma: Tensor


@dataclass(frozen=True)
class UCLRegularizationConfig:
    """Regularizer controls from the original PPO-UCL implementation."""

    beta: float = 0.03
    rho_reference: float = -2.783
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.beta < 0:
            raise ValueError("UCL beta must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("UCL epsilon must be positive")

    @property
    def std_reference(self) -> float:
        return math.log1p(math.exp(self.rho_reference))


def positive_sigma(rho: Tensor) -> Tensor:
    return torch.nn.functional.softplus(rho)


def snapshot_ucl_layers(
    layers: Sequence[UCLBayesianLayer],
) -> tuple[UCLLayerSnapshot, ...]:
    if not layers:
        raise ValueError("UCL snapshot requires at least one Bayesian layer")
    snapshots: list[UCLLayerSnapshot] = []
    for layer in layers:
        snapshots.append(
            UCLLayerSnapshot(
                weight_mu=layer.weight_mu.detach().clone(),
                weight_sigma=positive_sigma(layer.weight_rho).detach().clone(),
                bias_mu=layer.bias_mu.detach().clone(),
                bias_sigma=positive_sigma(layer.bias_rho).detach().clone(),
            )
        )
    return tuple(snapshots)


def ucl_regularization(
    layers: Sequence[UCLBayesianLayer],
    snapshots: Sequence[UCLLayerSnapshot],
    *,
    minibatch_size: int,
    config: UCLRegularizationConfig,
    saved_task: bool,
) -> dict[str, Tensor]:
    """Compute the original UCL mean/L1/sigma posterior penalties.

    The previous-layer uncertainty enters the current layer's L2 denominator,
    matching the uncertainty-guided connection-strength construction in the
    official PPO-UCL implementation.
    """

    config.validate()
    if minibatch_size < 1:
        raise ValueError("UCL minibatch_size must be positive")
    if len(layers) != len(snapshots) or not layers:
        raise ValueError("UCL layers and snapshots must be non-empty and aligned")

    device = layers[0].weight_mu.device
    dtype = layers[0].weight_mu.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    mean_l2 = zero.clone()
    mean_l1 = zero.clone()
    sigma_penalty = zero.clone()
    std_reference_sq = config.std_reference**2

    first_in = layers[0].in_features
    previous_weight_sigma = torch.nn.functional.softplus(
        torch.ones(
            (1, first_in),
            device=device,
            dtype=dtype,
        )
    )

    for layer, saved in zip(layers, snapshots, strict=True):
        current_weight_sigma_row = positive_sigma(layer.weight_rho).clamp_min(
            config.epsilon
        )
        current_bias_sigma = positive_sigma(layer.bias_rho).clamp_min(
            config.epsilon
        )
        saved_weight_sigma_row = saved.weight_sigma.to(
            device=device,
            dtype=dtype,
        ).clamp_min(config.epsilon)
        saved_bias_sigma = saved.bias_sigma.to(
            device=device,
            dtype=dtype,
        ).clamp_min(config.epsilon)
        saved_weight_mu = saved.weight_mu.to(device=device, dtype=dtype)
        saved_bias_mu = saved.bias_mu.to(device=device, dtype=dtype)

        out_features, in_features = layer.weight_mu.shape
        saved_weight_sigma = saved_weight_sigma_row.expand(
            out_features,
            in_features,
        )
        if previous_weight_sigma.shape[1] != in_features:
            raise ValueError(
                "UCL adjacent Bayesian layer widths are not chain-compatible"
            )
        previous_sigma = previous_weight_sigma.expand(
            out_features,
            in_features,
        )
        l2_sigma = torch.minimum(
            saved_weight_sigma,
            previous_sigma,
        ).clamp_min(config.epsilon)

        weight_delta = layer.weight_mu - saved_weight_mu
        bias_delta = layer.bias_mu - saved_bias_mu
        mean_l2 = mean_l2 + std_reference_sq * (
            (weight_delta / l2_sigma).square().sum()
            + (bias_delta / saved_bias_sigma).square().sum()
        )
        mean_l1 = mean_l1 + std_reference_sq * (
            (
                saved_weight_mu.square()
                / saved_weight_sigma.square()
                * weight_delta
            )
            .abs()
            .sum()
            + (
                saved_bias_mu.square()
                / saved_bias_sigma.square()
                * bias_delta
            )
            .abs()
            .sum()
        )

        weight_ratio = (
            current_weight_sigma_row.square()
            / saved_weight_sigma_row.square()
        ).clamp_min(config.epsilon)
        bias_ratio = (
            current_bias_sigma.square()
            / saved_bias_sigma.square()
        ).clamp_min(config.epsilon)
        current_weight_variance = current_weight_sigma_row.square().clamp_min(
            config.epsilon
        )
        current_bias_variance = current_bias_sigma.square().clamp_min(
            config.epsilon
        )
        sigma_penalty = sigma_penalty + (
            weight_ratio - torch.log(weight_ratio)
        ).sum()
        sigma_penalty = sigma_penalty + (
            current_weight_variance - torch.log(current_weight_variance)
        ).sum()
        sigma_penalty = sigma_penalty + (
            bias_ratio - torch.log(bias_ratio)
        ).sum()
        sigma_penalty = sigma_penalty + (
            current_bias_variance - torch.log(current_bias_variance)
        ).sum()

        previous_weight_sigma = saved_weight_sigma_row.reshape(1, -1)

    batch = float(minibatch_size)
    l2_term = mean_l2 / (2.0 * batch)
    l1_term = mean_l1 / batch if saved_task else zero.clone()
    sigma_term = config.beta * sigma_penalty / (2.0 * batch)
    total = l2_term + l1_term + sigma_term
    return {
        "total": total,
        "mean_l2": l2_term,
        "mean_l1": l1_term,
        "sigma": sigma_term,
    }
