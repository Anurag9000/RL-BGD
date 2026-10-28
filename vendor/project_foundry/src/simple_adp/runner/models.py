"""Model-route registry and adapter for the repository's estimator family."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Protocol, runtime_checkable

from ..baselines import vanilla_mlp_grid
from ..config import SearchSettings
from ..estimator import TabularADP
from ..registry import MODEL_REGISTRY
from ..tasks import ClassificationTask, RegressionTask
from .artifacts import estimator_model_path
from .config import RunnerSettings
from .datasets import TabularTaskData
from .route import ModelRoute, SuiteGroup


@runtime_checkable
class ModelAdapter(Protocol):
    """Plugin contract for translating registered model routes into predictions."""

    name: str

    def enumerate_routes(self, generic_profiles: tuple[str, ...], settings: RunnerSettings | None = None) -> list[ModelRoute]: ...

    def fit_predict(self, route: ModelRoute, *, data: TabularTaskData, split: Any, seed: int, settings: SearchSettings, checkpoint_path: str | None = None, progress_callback: Callable[[dict[str, object]], None] | None = None) -> dict[str, Any]: ...


# Public plan terminology: an adapter resolves a route and runs one child job.
RunAdapter = ModelAdapter


class ModelRegistry:
    """Explicit registry; duplicate names fail instead of silently replacing."""

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        name = adapter.name.strip()
        if not name:
            raise ValueError("model adapter name must be nonempty")
        if name in self._adapters:
            raise ValueError(f"model adapter {name!r} is already registered")
        self._adapters[name] = adapter

    def get(self, name: str) -> ModelAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise KeyError(f"unregistered model adapter {name!r}") from exc

    @property
    def adapters(self):
        return MappingProxyType(self._adapters)


class SimpleADPModelAdapter:
    """Run existing ``TabularADP`` implementations without modifying algorithms."""

    name = "simple_adp"

    @staticmethod
    def _take_rows(values: Any, row_indices: Any) -> Any:
        """Select split rows from pandas objects or array-backed providers.

        OpenML returns pandas objects, while custom ``DatasetProvider``
        implementations may expose NumPy arrays. Keep pandas' positional
        indexing semantics and use an integer index array for the generic
        tabular array path.
        """
        import numpy as np

        indices = np.asarray(row_indices, dtype=np.int64)
        if hasattr(values, "iloc"):
            return values.iloc[indices]
        return np.asarray(values)[indices]

    def enumerate_routes(
        self,
        generic_profiles: tuple[str, ...] = ("faithful", "corrected"),
        settings: RunnerSettings | None = None,
    ) -> list[ModelRoute]:
        if not generic_profiles:
            raise ValueError("at least one generic search profile is required")
        if set(generic_profiles) - {"faithful", "corrected"}:
            raise ValueError("generic profiles must be faithful or corrected")
        settings = settings or RunnerSettings()
        routes: list[ModelRoute] = []
        for model_id, config in MODEL_REGISTRY.items():
            suite_group = {
                "width": "width_only",
                "width_then_depth": "width_to_depth",
                "depth_then_width": "depth_to_width",
                "depth": "depth_only",
                "fixed": "vanilla",
            }.get(config.order, "unknown")
            if model_id == "MLP-Fixed":
                grid = tuple(tuple(widths) for widths in vanilla_mlp_grid())
                routes.append(ModelRoute(model_id, fixed_grid=grid, suite_group="vanilla"))
            elif config.order == "width":
                routes.extend(
                    ModelRoute(
                        model_id, label=f"{model_id}@depth-{depth}-width-{settings.width_only_width}",
                        suite_group=suite_group, init_width=settings.width_only_width, init_depth=depth,
                    )
                    for depth in settings.width_only_depths
                )
            elif config.order == "depth":
                routes.extend(
                    ModelRoute(
                        model_id, label=f"{model_id}@depth-1-width-{width}",
                        suite_group=suite_group, init_width=width, init_depth=1,
                    )
                    for width in settings.depth_only_widths
                )
            elif config.generic:
                routes.extend(
                    ModelRoute(
                        model_id,
                        generic_profile=profile,
                        label=f"{model_id}@{profile}@depth-{settings.width_to_depth_depth}-width-{settings.width_to_depth_width}",
                        suite_group=suite_group,
                        init_width=settings.width_to_depth_width,
                        init_depth=settings.width_to_depth_depth,
                    )
                    for profile in generic_profiles
                )
            else:
                routes.append(ModelRoute(
                    model_id,
                    label=f"{model_id}@depth-{settings.width_to_depth_depth}-width-{settings.width_to_depth_width}",
                    suite_group=suite_group,
                    init_width=settings.width_to_depth_width,
                    init_depth=settings.width_to_depth_depth,
                ))
        routes.extend(
            ModelRoute(
                "MLP-Fixed",
                fixed_grid=(tuple(widths),),
                label=f"vanilla-d{len(widths)}-w{widths[0]}",
                suite_group="vanilla",
            )
            for widths in vanilla_mlp_grid()
        )
        if settings.run_hardness_screen:
            architectures = [(1, width) for width in settings.screen_widths]
            architectures.extend(
                (depth, width)
                for depth in settings.screen_depths
                for width in settings.deeper_screen_widths
            )
            routes.extend(
                ModelRoute(
                    "MLP-Fixed",
                    fixed_grid=(tuple([width] * depth),),
                    label=f"screen-d{depth}-w{width}-seed{settings.seed + seed_index}",
                    suite_group="screening",
                    init_width=width,
                    init_depth=depth,
                    screening_only=True,
                    screening_seed=settings.seed + seed_index,
                    screening_architecture=(depth, width),
                )
                for depth, width in architectures
                for seed_index in range(settings.screen_seed_count)
            )
        return routes

    def fit_predict(
        self,
        route: ModelRoute,
        *,
        data: TabularTaskData,
        split: Any,
        seed: int,
        settings: SearchSettings,
        checkpoint_path: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, Any]:
        import time

        take_rows = self._take_rows
        X_train = take_rows(data.X, split.train_indices)
        y_train = take_rows(data.y, split.train_indices)
        X_valid = take_rows(data.X, split.validation_indices)
        y_valid = take_rows(data.y, split.validation_indices)
        if not route.screening_only:
            X_test = take_rows(data.X, split.test_indices)
            y_test = take_rows(data.y, split.test_indices)
        task = getattr(data, "task_adapter", None)
        if task is None:
            task_kind = getattr(data, "task_kind", getattr(getattr(data, "suite", None), "task_kind", None))
            if task_kind == "classification":
                task = ClassificationTask()
            elif task_kind == "regression":
                task = RegressionTask()
            else:
                raise ValueError("custom task data must expose task_adapter or classification/regression task_kind")
        effective_settings = SearchSettings(**{
            **settings.as_dict(),
            "init_width": route.init_width or settings.init_width,
            "init_depth": route.init_depth or settings.init_depth,
            "seed": int(seed),
        })
        estimator = TabularADP(
            route.model_id,
            task=task,
            settings=effective_settings,
            generic_profile=route.generic_profile or "faithful",
            feature_schema=data.feature_schema,
            fixed_grid=[list(widths) for widths in route.fixed_grid] if route.fixed_grid is not None else None,
        )
        started = time.perf_counter()
        estimator.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            checkpoint_path=checkpoint_path,
            progress_callback=progress_callback,
        )
        fit_seconds = time.perf_counter() - started
        validation_outputs = estimator._outputs(X_valid)
        validation_targets = estimator.task.encode_targets(y_valid, device=validation_outputs.device)
        validation_score = estimator.task.validation_score(validation_outputs, validation_targets)
        task_kind = getattr(data, "task_kind", getattr(getattr(data, "suite", None), "task_kind", None))
        if route.screening_only:
            validation_prediction = estimator.predict(X_valid)
            if task_kind == "classification":
                validation_metrics = task.evaluation_metrics(
                    y_valid, validation_prediction, estimator.predict_proba(X_valid),
                )
                validation_metrics = {
                    key: value for key, value in validation_metrics.items()
                    if key in {"accuracy", "balanced_accuracy"}
                }
            elif task_kind == "regression":
                validation_metrics = {"r2": task.evaluation_metrics(y_valid, validation_prediction).get("r2")}
            else:
                validation_metrics = {"selection_score": float(validation_score)}
            return {
                "fit_seconds": fit_seconds,
                "history_entries": len(estimator.history_),
                "selected_architecture": list(estimator.model_.hidden_widths),
                "validation_selection_score": float(validation_score),
                "validation_metrics": validation_metrics,
                "test_metrics": {},
                "screening_only": True,
                "screening_seed": route.screening_seed,
                "screening_architecture": list(route.screening_architecture or ()),
                "artifact_path": None,
            }
        prediction = estimator.predict(X_test)
        metrics: dict[str, float | None]
        if task_kind == "classification":
            probabilities = estimator.predict_proba(X_test)
            metrics = task.evaluation_metrics(y_test, prediction, probabilities)
        elif task_kind == "regression":
            metrics = task.evaluation_metrics(y_test, prediction)
        else:
            try:
                probabilities = estimator.predict_proba(X_test)
            except (TypeError, NotImplementedError):
                probabilities = None
            metrics = task.evaluation_metrics(y_test, prediction, probabilities)
        artifact_path = None
        if checkpoint_path is not None:
            artifact_path = str(estimator_model_path(checkpoint_path))
            estimator.save(artifact_path)
        return {
            "fit_seconds": fit_seconds,
            "history_entries": len(estimator.history_),
            "selected_architecture": list(estimator.model_.hidden_widths),
            "validation_selection_score": float(validation_score),
            "test_metrics": metrics,
            "artifact_path": artifact_path,
        }
