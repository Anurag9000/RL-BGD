"""Reusable pressure-aware experiment-suite orchestration for simple-adp.

Runner types load on first access. This keeps the worker entrypoint light enough
to initialize optional dataframe/ML acceleration before importing estimators.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RunnerSettings": "config",
    "StandalonePressureSettings": "config",
    "DatasetProvider": "datasets",
    "DatasetRegistry": "datasets",
    "DatasetTaskRef": "datasets",
    "OpenMLDatasetProvider": "datasets",
    "TabularTaskData": "datasets",
    "ModelAdapter": "models",
    "ModelRegistry": "models",
    "ModelRoute": "route",
    "RunAdapter": "models",
    "SimpleADPModelAdapter": "models",
    "SuiteGroup": "route",
    "RunIdentity": "plan",
    "RunSpec": "plan",
    "SplitSpec": "plan",
    "SuitePlan": "plan",
    "CallbackResourceProvider": "resources",
    "ResourceProvider": "resources",
    "SuiteCoordinator": "scheduler",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
