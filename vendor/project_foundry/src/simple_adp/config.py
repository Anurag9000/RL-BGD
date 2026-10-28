"""Immutable configuration types for canonical and generic ADP searches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ControllerName = Literal[
    "lgnd",
    "width_den",
    "depth_den",
    "phase_search",
    "fixed",
]
WidthStrategy = Literal["all_layers", "round_robin"]
SearchOrder = Literal["width", "depth", "width_then_depth", "depth_then_width", "fixed"]
DepthSeed = Literal["one", "current"]
GenericProfile = Literal["faithful", "corrected"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Algorithm identity and defaults; runtime limits can be overridden per estimator."""

    id: str
    controller: ControllerName
    order: SearchOrder
    width_strategy: WidthStrategy = "all_layers"
    depth_seed: DepthSeed = "one"
    progressive_depth: bool = False
    generic: bool = False
    optimizer_profile: Literal["adam_cosine", "adamw"] | None = None
    patience_width_exp: int = 10
    patience_depth_exp: int = 10
    delta_width: float = 0.0
    delta_depth: float = 0.0
    max_width: int = 512
    max_depth: int = 5
    max_neurons: int = 1_000_000_000
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("model id must be non-empty")
        if self.controller == "fixed" and self.order != "fixed":
            raise ValueError("fixed controller must use fixed search order")
        if self.controller != "fixed" and self.order == "fixed":
            raise ValueError("adaptive controller cannot use fixed search order")
        if self.optimizer_profile is None:
            object.__setattr__(self, "optimizer_profile", "adamw" if self.generic else "adam_cosine")
        if self.patience_width_exp < 0 or self.patience_depth_exp < 0:
            raise ValueError("architecture patience cannot be negative")
        if self.delta_width < 0 or self.delta_depth < 0:
            raise ValueError("improvement margins cannot be negative")
        if min(self.max_width, self.max_depth, self.max_neurons) <= 0:
            raise ValueError("architecture capacity limits must be positive")

    def signature_dict(self) -> dict[str, object]:
        """Return every mechanism/default field that can affect a search.

        The human-readable description is intentionally omitted. A checkpoint
        must become incompatible when a canonical ID is remapped to a different
        controller or growth policy, even if the ID itself stays the same.
        """
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "description"
        }


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """Per-run values for stage optimization, search, and capacity limits."""

    init_width: int = 16
    init_depth: int = 2
    ex_k_width: int = 1
    # Recorded API knob; dedicated depth expansion still appends one layer per call.
    ex_k_depth: int = 1
    patience_es: int = 10
    patience_width_exp: int = 10
    patience_depth_exp: int = 10
    delta_width: float = 0.0
    delta_depth: float = 0.0
    max_width: int = 512
    max_depth: int = 5
    max_neurons: int = 1_000_000_000
    max_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    scheduler_t0: int = 10
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float | None = None
    # adp_search loss weights.  The guide does not prescribe defaults; simple-adp
    # uses prediction-only loss unless an adapter penalty weight is enabled.
    lambda_loss: float = 1.0
    lambda_eq: float = 0.0
    lambda_ineq: float = 0.0
    max_candidates: int = 0
    min_new_layer_width: int = 1
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = {
            "init_width": self.init_width,
            "init_depth": self.init_depth,
            "ex_k_width": self.ex_k_width,
            "ex_k_depth": self.ex_k_depth,
            "patience_es": self.patience_es,
            "max_width": self.max_width,
            "max_depth": self.max_depth,
            "max_neurons": self.max_neurons,
            "min_new_layer_width": self.min_new_layer_width,
            "batch_size": self.batch_size,
            "scheduler_t0": self.scheduler_t0,
            "scheduler_t_mult": self.scheduler_t_mult,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"settings must be positive: {', '.join(bad)}")
        if self.patience_width_exp < 0 or self.patience_depth_exp < 0:
            raise ValueError("architecture patience cannot be negative")
        if self.delta_width < 0 or self.delta_depth < 0:
            raise ValueError("improvement margins cannot be negative")
        if (self.learning_rate <= 0 or self.weight_decay < 0 or self.scheduler_eta_min < 0
                or min(self.lambda_loss, self.lambda_eq, self.lambda_ineq) < 0):
            raise ValueError("invalid optimizer settings")
        if self.max_epochs < 0:
            raise ValueError("max_epochs cannot be negative; zero means unlimited")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive when set")
        if self.max_candidates < 0:
            raise ValueError("max_candidates cannot be negative")

    def as_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
