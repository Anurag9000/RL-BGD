"""Shared results returned by architecture-search controllers."""

from dataclasses import dataclass, field

from .models import TabularMLP


@dataclass
class SearchResult:
    model: TabularMLP
    score: float
    history: list[dict[str, object]] = field(default_factory=list)
    state: dict[str, object] = field(default_factory=dict)
