"""Lightweight, serializable model route identities.

Keeping route data separate from estimator adapters lets a suite worker inspect
its persisted job and initialize backend hooks before importing model libraries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

SuiteGroup = Literal["width_only", "width_to_depth", "depth_to_width", "depth_only", "vanilla", "screening"]


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """One uniquely labeled model/profile/architecture route."""

    model_id: str
    generic_profile: str | None = None
    fixed_grid: tuple[tuple[int, ...], ...] | None = None
    label: str | None = None
    init_width: int | None = None
    init_depth: int | None = None
    screening_only: bool = False
    screening_seed: int | None = None
    screening_architecture: tuple[int, int] | None = None
    suite_group: SuiteGroup | None = None

    @property
    def route_label(self) -> str:
        if self.label:
            return self.label
        return self.model_id if self.generic_profile is None else f"{self.model_id}@{self.generic_profile}"

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["fixed_grid"] = [list(widths) for widths in self.fixed_grid] if self.fixed_grid is not None else None
        value["route_label"] = self.route_label
        return value
