"""Load optional provider/model plugins through Python package entry points."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from .datasets import DatasetRegistry, OpenMLDatasetProvider
from .models import ModelRegistry, SimpleADPModelAdapter


def _instantiate(value: Any) -> Any:
    return value() if isinstance(value, type) else value


def load_registries() -> tuple[DatasetRegistry, ModelRegistry]:
    datasets = DatasetRegistry()
    models = ModelRegistry()
    datasets.register(OpenMLDatasetProvider())
    models.register(SimpleADPModelAdapter())
    points = entry_points()
    for group, registry in (
        ("simple_adp.dataset_providers", datasets),
        ("simple_adp.model_adapters", models),
    ):
        for point in sorted(points.select(group=group), key=lambda entry: entry.name):
            registry.register(_instantiate(point.load()))
    return datasets, models
