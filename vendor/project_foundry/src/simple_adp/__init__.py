"""Modular adaptive architecture search for tabular MLPs.

Public names are loaded lazily so importing a suite worker does not import
pandas or model adapters before its optional backend hooks are initialized.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CheckpointStore": "checkpoint",
    "capture_rng_state": "checkpoint",
    "restore_rng_state": "checkpoint",
    "stable_signature": "checkpoint",
    "OPENML_SUITES": "benchmark",
    "OpenMLBenchmarkRunner": "benchmark",
    "load_openml_task": "benchmark",
    "load_suite_tasks": "benchmark",
    "make_fold_split": "benchmark",
    "vanilla_mlp_grid": "baselines",
    "GenericProfile": "config",
    "ModelConfig": "config",
    "SearchSettings": "config",
    "DedicatedController": "dedicated",
    "TabularADP": "estimator",
    "GenericCandidateSearch": "generic_search",
    "can_add_depth": "growth",
    "can_grow_width_all_layers": "growth",
    "can_grow_width_round_robin": "growth",
    "expand_depth": "growth",
    "expand_depth_match_width": "growth",
    "expand_width": "growth",
    "expand_width_all_layers": "growth",
    "expand_width_den_round_robin": "growth",
    "expand_width_round_robin": "growth",
    "scalar_width": "growth",
    "total_neurons": "growth",
    "TabularMLP": "models",
    "rebuild_mlp": "models",
    "FeatureColumn": "preprocessing",
    "TabularPreprocessor": "preprocessing",
    "MODEL_REGISTRY": "registry",
    "get_model_config": "registry",
    "SearchController": "search",
    "SearchResult": "search",
    "CallableTask": "tasks",
    "ClassificationTask": "tasks",
    "RegressionTask": "tasks",
    "TaskAdapter": "tasks",
    "TaskOutputSpec": "tasks",
    "StageResult": "training",
    "StageTrainer": "training",
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
