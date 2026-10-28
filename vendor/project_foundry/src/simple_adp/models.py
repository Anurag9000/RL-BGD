"""Feed-forward MLP and architecture-aware parameter transfer."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn


class TabularMLP(nn.Module):
    """MLP whose hidden architecture is an explicit ordered width vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_widths: Sequence[int],
        output_dim: int,
        *,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if not hidden_widths or any(int(width) <= 0 for width in hidden_widths):
            raise ValueError("hidden_widths must contain positive widths")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.batch_norm = bool(batch_norm)
        self.hidden_widths = [int(width) for width in hidden_widths]
        layers: list[nn.Module] = []
        in_features = self.input_dim
        for width in self.hidden_widths:
            layers.append(nn.Linear(in_features, width))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            in_features = width
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_features, self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = features
        for layer in self.hidden_layers:
            value = layer(value)
        return self.output_layer(value)

    @property
    def depth(self) -> int:
        return len(self.hidden_widths)

    @property
    def width(self) -> int:
        return min(self.hidden_widths)

    def snapshot(self, *, cursor: int = 0) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_widths": list(self.hidden_widths),
            "batch_norm": self.batch_norm,
            "cursor": int(cursor),
            "state_dict": copy.deepcopy(self.state_dict()),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> "TabularMLP":
        model = cls(
            int(snapshot["input_dim"]),
            snapshot["hidden_widths"],  # type: ignore[arg-type]
            int(snapshot["output_dim"]),
            batch_norm=bool(snapshot.get("batch_norm", False)),
        )
        model.load_state_dict(snapshot["state_dict"])  # type: ignore[arg-type]
        setattr(model, "_width_cursor", int(snapshot.get("cursor", 0)))
        return model


def _copy_overlap(source: torch.Tensor, destination: torch.Tensor) -> None:
    slices = tuple(slice(0, min(a, b)) for a, b in zip(source.shape, destination.shape))
    if len(slices) == source.ndim:
        destination[slices].copy_(source[slices])


def rebuild_mlp(
    model: TabularMLP,
    hidden_widths: Sequence[int],
    *,
    cursor: int = 0,
) -> TabularMLP:
    """Rebuild at new widths, copying overlapping tensors and leaving new cells initialized."""
    new_model = TabularMLP(
        model.input_dim,
        hidden_widths,
        model.output_dim,
        batch_norm=model.batch_norm,
    ).to(next(model.parameters()).device)
    old_state = model.state_dict()
    new_state = new_model.state_dict()
    with torch.no_grad():
        for key, value in new_state.items():
            previous = old_state.get(key)
            if previous is not None and previous.ndim == value.ndim:
                _copy_overlap(previous, value)
        new_model.load_state_dict(new_state)
    setattr(new_model, "_width_cursor", int(cursor))
    return new_model


def from_snapshot_preserving_rng(snapshot: dict[str, object]) -> TabularMLP:
    """Restore a saved model without shifting later architecture initialization.

    Constructors initialize parameters before the saved state is loaded. Search
    controllers use this helper when materializing the global-best snapshot so
    that bookkeeping does not consume RNG draws that belong to later growth
    candidates. Resume reconstruction uses the same rule.
    """
    state = torch.get_rng_state()
    try:
        return TabularMLP.from_snapshot(snapshot)
    finally:
        torch.set_rng_state(state)
