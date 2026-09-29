"""Small reusable MLP builder."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


class MLP(nn.Sequential):
    """Feed-forward MLP with configurable hidden widths and activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
        output_activation: nn.Module | None = None,
    ) -> None:
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(activation())
        if output_activation is not None:
            layers.append(output_activation)
        super().__init__(*layers)
