"""Posterior interfaces shared by Bayesian update engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn


class ParameterPosterior(ABC):
    """Abstract posterior over a module's trainable parameters."""

    @abstractmethod
    def sample(self, *, antithetic: bool = False, samples: int = 1) -> list[dict[str, Tensor]]:
        """Draw parameter dictionaries suitable for stateless module evaluation."""

    @abstractmethod
    def mean_parameters(self) -> dict[str, Tensor]:
        """Return posterior mean parameters."""

    @abstractmethod
    def sync_module(self, module: nn.Module) -> None:
        """Copy posterior means into a module in place."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable posterior state."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore posterior state."""
