"""Reproducible OpenML benchmark loading, hardness screening, and evaluation.

OpenML and scikit-learn are optional dependencies, imported only when this
module's runner is used. Dataset eligibility is decided on inner validation
data only; outer test rows are never used for screening or model selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from .baselines import vanilla_mlp_grid
from .config import SearchSettings
from .estimator import TabularADP
from .registry import MODEL_REGISTRY
from .tasks import ClassificationTask, RegressionTask

TaskKind = Literal["classification", "regression"]


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    key: str
    openml_id: int
    task_kind: TaskKind
    description: str
    hardness_source: str


OPENML_SUITES = {
    "cc18": SuiteSpec(
        "cc18", 99, "classification", "OpenML-CC18 Curated Classification benchmark",
        "Broad curated IID classification pool; use validation screening to remove tasks solved by shallow MLPs.",
    ),
    "tabzilla": SuiteSpec(
        "tabzilla", 379, "classification", "TabZilla Hard Datasets",
        "Selected as the 36 hardest classification datasets in the TabZilla study; still screen for shallow-MLP saturation.",
    ),
    "ctr23": SuiteSpec(
        "ctr23", 353, "regression", "OpenML-CTR23 curated tabular regression suite",
        "Curated IID regression tasks, including a criterion excluding tasks exactly solved by a linear model.",
    ),
}


@dataclass(frozen=True, slots=True)
class FoldSplit:
    repeat: int
    fold: int
    sample: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    split_sha256: str


@dataclass(slots=True)
class TaskData:
    suite: SuiteSpec
    task_id: int
    dataset_id: int
    name: str
    target_name: str
    X: pd.DataFrame
    y: pd.Series
    feature_schema: list[str]
    dataset_metadata: dict[str, Any]
    content_sha256: str
    task: Any

    @property
    def task_kind(self) -> TaskKind:
        """Expose provider task semantics through the generic task-data API."""
        return self.suite.task_kind

    @property
    def task_adapter(self):
        """Build a fresh mutable target adapter for each worker route."""
        return _task_adapter(self.task_kind)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    return repr(value)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, default=_json_default, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()


def _sha_indices(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype=np.int64)
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sha_frame(frame: pd.DataFrame, target: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([str(name) for name in frame.columns]).encode())
    digest.update(json.dumps({str(name): str(dtype) for name, dtype in frame.dtypes.items()}, sort_keys=True).encode())
    try:
        digest.update(pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy().tobytes())
        digest.update(pd.util.hash_pandas_object(target, index=False, categorize=True).to_numpy().tobytes())
    except (TypeError, ValueError):
        digest.update(frame.to_json(orient="split", default_handler=repr).encode())
        digest.update(target.to_json(orient="values", default_handler=repr).encode())
    return digest.hexdigest()


def load_suite_tasks(suite_key: str) -> list[int]:
    """Fetch all task IDs in an official OpenML suite."""
    if suite_key not in OPENML_SUITES:
        raise ValueError(f"unknown suite {suite_key!r}; choose from {', '.join(OPENML_SUITES)}")
    try:
        import openml
    except ImportError as exc:
        raise RuntimeError("benchmark loading requires the optional dependencies: pip install 'simple-adp[benchmark]'") from exc
    spec = OPENML_SUITES[suite_key]
    suite = openml.study.get_suite(spec.openml_id)
    return [int(task_id) for task_id in suite.tasks]


def load_openml_task(suite_key: str, task_id: int) -> TaskData:
    """Download one task's features and target through the OpenML task API."""
    try:
        import openml
    except ImportError as exc:
        raise RuntimeError("benchmark loading requires the optional dependencies: pip install 'simple-adp[benchmark]'") from exc
    spec = OPENML_SUITES[suite_key]
    task = openml.tasks.get_task(int(task_id))
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(target=task.target_name)
    if y is None:
        raise ValueError(f"OpenML task {task_id} did not return a target")
    for column in X.columns:
        if isinstance(X[column].dtype, pd.SparseDtype):
            X[column] = X[column].sparse.to_dense()
    return TaskData(
        suite=spec,
        task_id=int(task_id),
        dataset_id=int(task.dataset_id),
        name=str(dataset.name),
        target_name=str(task.target_name),
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        feature_schema=["categorical" if is_categorical else "numeric" for is_categorical in categorical_indicator],
        dataset_metadata={
            "license": getattr(dataset, "licence", None),
            "citation": getattr(dataset, "citation", None),
            "source_url": getattr(dataset, "original_data_url", None),
            "dataset_url": getattr(dataset, "url", None),
            "version": getattr(dataset, "version", None),
            "md5_checksum": getattr(dataset, "md5_checksum", None),
        },
        content_sha256=_sha_frame(X, y),
        task=task,
    )


def make_fold_split(task: Any, y: pd.Series, suite: SuiteSpec, *, repeat: int, fold: int, sample: int, validation_fraction: float, split_seed: int) -> FoldSplit:
    """Keep OpenML test rows fixed and split only its training rows for validation."""
    from sklearn.model_selection import train_test_split

    outer_train, test = task.get_train_test_split_indices(repeat=repeat, fold=fold, sample=sample)
    outer_train = np.asarray(outer_train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    labels = y.iloc[outer_train].to_numpy()
    stratify = None
    if suite.task_kind == "classification":
        counts = pd.Series(labels).value_counts(dropna=False)
        if len(counts) < 2 or int(counts.min()) < 2:
            raise ValueError("outer training partition cannot support stratified train/validation splitting")
        stratify = labels
    train, valid = train_test_split(
        outer_train,
        test_size=validation_fraction,
        random_state=split_seed,
        shuffle=True,
        stratify=stratify,
    )
    train = np.asarray(train, dtype=np.int64)
    valid = np.asarray(valid, dtype=np.int64)
    if set(train).intersection(valid) or set(train).intersection(test) or set(valid).intersection(test):
        raise RuntimeError("generated split contains overlapping rows")
    return FoldSplit(repeat, fold, sample, train, valid, test, _sha_indices(train, valid, test))


def _task_adapter(kind: TaskKind):
    return ClassificationTask() if kind == "classification" else RegressionTask()


def _classification_metrics(y_true, prediction, probability, labels) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probability, labels=list(labels))),
    }


def _regression_metrics(y_true, prediction) -> dict[str, float | None]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    r2 = float(r2_score(y_true, prediction))
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": r2 if math.isfinite(r2) else None,
    }


def model_runs(profiles: Iterable[str]) -> list[tuple[str, str | None, list[list[int]] | None]]:
    """Return canonical runs plus each individual vanilla MLP candidate."""
    runs: list[tuple[str, str | None, list[list[int]] | None]] = []
    for model_id, config in MODEL_REGISTRY.items():
        if model_id == "MLP-Fixed":
            runs.append((model_id, None, vanilla_mlp_grid()))
        elif config.generic:
            for profile in profiles:
                runs.append((model_id, profile, None))
        else:
            runs.append((model_id, None, None))
    # The aggregate MLP-Fixed run selects among all vanilla candidates. These
    # additional one-candidate runs expose every vanilla architecture's score.
    for widths in vanilla_mlp_grid():
        label = "vanilla-d{}-w{}".format(len(widths), widths[0])
        runs.append((label, None, [widths]))
    return runs


class OpenMLBenchmarkRunner:
    """Run a validation-screened, task-split-respecting ADP/vanilla benchmark."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        settings: SearchSettings | None = None,
        suites: Iterable[str] = ("cc18", "tabzilla", "ctr23"),
        profiles: Iterable[str] = ("faithful", "corrected"),
        seeds: Iterable[int] = (0, 1, 2),
        validation_fraction: float = 0.2,
        screen_widths: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        screen_depths: Iterable[int] = (2, 3, 5),
        deeper_screen_widths: Iterable[int] = (16, 64, 256, 512),
        classification_ceiling: float = 1.0,
        regression_ceiling_r2: float = 0.999,
        min_complexity_gain: float = 0.02,
        min_hard_fold_fraction: float = 0.8,
        screen_seed_count: int = 2,
        task_limit: int | None = None,
        fold_limit: int | None = None,
        screen_only: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.settings = settings or SearchSettings()
        self.suites = tuple(suites)
        self.profiles = tuple(profiles)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.validation_fraction = validation_fraction
        self.screen_widths = tuple(int(width) for width in screen_widths)
        self.screen_depths = tuple(int(depth) for depth in screen_depths)
        self.deeper_screen_widths = tuple(int(width) for width in deeper_screen_widths)
        self.classification_ceiling = classification_ceiling
        self.regression_ceiling_r2 = regression_ceiling_r2
        self.min_complexity_gain = min_complexity_gain
        self.min_hard_fold_fraction = min_hard_fold_fraction
        self.screen_seed_count = screen_seed_count
        self.task_limit = task_limit
        self.fold_limit = fold_limit
        self.screen_only = screen_only
        if not self.suites:
            raise ValueError("at least one OpenML suite is required")
        unknown_suites = set(self.suites) - set(OPENML_SUITES)
        if unknown_suites:
            raise ValueError(f"unknown suite(s): {', '.join(sorted(unknown_suites))}")
        if not self.seeds:
            raise ValueError("at least one full-run seed is required")
        if not self.profiles:
            raise ValueError("at least one generic-search profile is required")
        if not 0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if not self.screen_widths or any(width <= 0 for width in self.screen_widths):
            raise ValueError("screen_widths must contain positive hidden widths")
        if not self.screen_depths or any(depth <= 1 for depth in self.screen_depths):
            raise ValueError("screen_depths must contain depths greater than one")
        if not self.deeper_screen_widths or any(width <= 0 for width in self.deeper_screen_widths):
            raise ValueError("deeper_screen_widths must contain positive hidden widths")
        if screen_seed_count <= 0:
            raise ValueError("screen_seed_count must be positive")
        if min_complexity_gain < 0:
            raise ValueError("min_complexity_gain cannot be negative")
        if not 0 < min_hard_fold_fraction <= 1:
            raise ValueError("min_hard_fold_fraction must be in (0, 1]")
        if not 0 <= classification_ceiling <= 1:
            raise ValueError("classification_ceiling must be between 0 and 1")
        if task_limit is not None and task_limit <= 0:
            raise ValueError("task_limit must be positive")
        if fold_limit is not None and fold_limit <= 0:
            raise ValueError("fold_limit must be positive")
        if set(self.profiles) - {"faithful", "corrected"}:
            raise ValueError("profiles may contain only faithful and corrected")
        signature_payload = {
            "protocol_version": 2,
            "settings": self.settings.as_dict(), "suites": self.suites, "profiles": self.profiles,
            "seeds": self.seeds, "validation_fraction": validation_fraction,
            "screen_widths": self.screen_widths, "classification_ceiling": classification_ceiling,
            "screen_depths": self.screen_depths, "deeper_screen_widths": self.deeper_screen_widths,
            "regression_ceiling_r2": regression_ceiling_r2,
            "min_complexity_gain": min_complexity_gain, "min_hard_fold_fraction": min_hard_fold_fraction,
            "screen_seed_count": screen_seed_count,
        }
        self.run_signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, default=_json_default).encode()
        ).hexdigest()

    def _task_manifest(self, task_data: TaskData, dimensions: tuple[int, int, int]) -> dict[str, Any]:
        return {
            "suite": task_data.suite.key,
            "suite_openml_id": task_data.suite.openml_id,
            "suite_name": task_data.suite.description,
            "task_id": task_data.task_id,
            "dataset_id": task_data.dataset_id,
            "dataset_name": task_data.name,
            "task_kind": task_data.suite.task_kind,
            "target_name": task_data.target_name,
            "rows": int(len(task_data.X)),
            "raw_features": int(task_data.X.shape[1]),
            "feature_types": {str(name): str(dtype) for name, dtype in task_data.X.dtypes.items()},
            "openml_feature_schema": dict(zip(task_data.X.columns, task_data.feature_schema, strict=True)),
            "target_cardinality": int(task_data.y.nunique(dropna=False)),
            "provenance": task_data.dataset_metadata,
            "content_sha256": task_data.content_sha256,
            "split_dimensions_repeat_fold_sample": list(dimensions),
            "hardness_source": task_data.suite.hardness_source,
        }

    def _save_split(self, data: TaskData, split: FoldSplit) -> str:
        path = self.output_dir / "splits" / data.suite.key / f"task-{data.task_id}-r{split.repeat}-f{split.fold}-s{split.sample}-{self.run_signature[:12]}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temp = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temp,
                train=split.train_indices,
                validation=split.validation_indices,
                test=split.test_indices,
                split_sha256=np.asarray(split.split_sha256),
            )
            temp.replace(path)
        return str(path.relative_to(self.output_dir))

    def _screen_fold(self, data: TaskData, split: FoldSplit) -> dict[str, Any]:
        X_train = data.X.iloc[split.train_indices]
        y_train = data.y.iloc[split.train_indices]
        X_valid = data.X.iloc[split.validation_indices]
        y_valid = data.y.iloc[split.validation_indices]
        records: list[dict[str, Any]] = []
        saturated_by_model: dict[str, bool] = {}
        scores_by_model: dict[str, list[float]] = {}
        screen_architectures = [(1, width) for width in self.screen_widths]
        screen_architectures.extend(
            (depth, width)
            for depth in self.screen_depths
            for width in self.deeper_screen_widths
        )
        for depth, width in screen_architectures:
            model_label = f"d{depth}-w{width}"
            each_seed = []
            scores_by_model[model_label] = []
            for index in range(self.screen_seed_count):
                seed = self.settings.seed + index
                settings = SearchSettings(**{**self.settings.as_dict(), "seed": seed})
                estimator = TabularADP(
                    "MLP-Fixed", task=_task_adapter(data.suite.task_kind), settings=settings,
                    fixed_grid=[[width] * depth], feature_schema=data.feature_schema,
                ).fit(X_train, y_train, validation_data=(X_valid, y_valid))
                prediction = estimator.predict(X_valid)
                if data.suite.task_kind == "classification":
                    from sklearn.metrics import accuracy_score, balanced_accuracy_score

                    accuracy = float(accuracy_score(y_valid, prediction))
                    balanced = float(balanced_accuracy_score(y_valid, prediction))
                    score = balanced
                    is_ceiling = accuracy >= self.classification_ceiling
                    metric = {"accuracy": accuracy, "balanced_accuracy": balanced}
                else:
                    from sklearn.metrics import r2_score

                    raw_score = float(r2_score(y_valid, prediction))
                    score = raw_score if math.isfinite(raw_score) else -1.0
                    is_ceiling = score >= self.regression_ceiling_r2
                    metric = {"r2": raw_score if math.isfinite(raw_score) else None}
                each_seed.append(is_ceiling)
                scores_by_model[model_label].append(score)
                records.append({"architecture": model_label, "seed": seed, "metrics": metric, "ceiling": is_ceiling})
            saturated_by_model[model_label] = all(each_seed)
        # Exclude folds saturated by a one-layer MLP. The second gate requires
        # a measurable validation gain from depth before comparing the methods.
        shallow_saturated = any(saturated_by_model.get(f"d1-w{width}", False) for width in self.screen_widths)
        mean_scores = {name: float(np.mean(scores)) for name, scores in scores_by_model.items()}
        shallow_scores = [score for name, score in mean_scores.items() if name.startswith("d1-")]
        deep_scores = [score for name, score in mean_scores.items() if not name.startswith("d1-")]
        best_shallow = max(shallow_scores)
        best_deep = max(deep_scores)
        complexity_gain = best_deep - best_shallow
        complexity_sensitive = complexity_gain >= self.min_complexity_gain
        return {
            "saturated": shallow_saturated,
            "complexity_sensitive": complexity_sensitive,
            "complexity_metric": "balanced_accuracy" if data.suite.task_kind == "classification" else "r2",
            "best_shallow_validation_score": best_shallow,
            "best_deeper_validation_score": best_deep,
            "validation_complexity_gain": complexity_gain,
            "min_complexity_gain": self.min_complexity_gain,
            "min_hard_fold_fraction": self.min_hard_fold_fraction,
            "criterion": "shallow_ceiling_gate_plus_validation_depth_gain_gate",
            "shallow_ceiling_threshold": self.classification_ceiling if data.suite.task_kind == "classification" else self.regression_ceiling_r2,
            "screen_seed_count": self.screen_seed_count,
            "models": records,
        }

    def _run_one(self, data: TaskData, split: FoldSplit, model_id: str, profile: str | None, grid: list[list[int]] | None, seed: int) -> dict[str, Any]:
        X_train = data.X.iloc[split.train_indices]
        y_train = data.y.iloc[split.train_indices]
        X_valid = data.X.iloc[split.validation_indices]
        y_valid = data.y.iloc[split.validation_indices]
        X_test = data.X.iloc[split.test_indices]
        y_test = data.y.iloc[split.test_indices]
        settings = SearchSettings(**{**self.settings.as_dict(), "seed": seed})
        adapter = _task_adapter(data.suite.task_kind)
        started = time.perf_counter()
        estimator = TabularADP(
            model_id if not model_id.startswith("vanilla-") else "MLP-Fixed",
            task=adapter,
            settings=settings,
            generic_profile=profile or "faithful",
            feature_schema=data.feature_schema,
            fixed_grid=grid if (model_id == "MLP-Fixed" or model_id.startswith("vanilla-")) else None,
        ).fit(X_train, y_train, validation_data=(X_valid, y_valid))
        fit_seconds = time.perf_counter() - started
        validation_outputs = estimator._outputs(X_valid)
        validation_targets = estimator.task.encode_targets(y_valid, device=validation_outputs.device)
        validation_score = estimator.task.validation_score(validation_outputs, validation_targets)
        prediction = estimator.predict(X_test)
        if data.suite.task_kind == "classification":
            probability = estimator.predict_proba(X_test)
            metrics = _classification_metrics(y_test, prediction, probability, estimator.task.classes_)
        else:
            metrics = _regression_metrics(y_test, prediction)
        run_label = model_id if profile is None else f"{model_id}@{profile}"
        return {
            "suite": data.suite.key,
            "task_id": data.task_id,
            "dataset_id": data.dataset_id,
            "dataset_sha256": data.content_sha256,
            "dataset_name": data.name,
            "repeat": split.repeat,
            "fold": split.fold,
            "sample": split.sample,
            "split_sha256": split.split_sha256,
            "model": run_label,
            "model_id": model_id if model_id in MODEL_REGISTRY else "MLP-Fixed",
            "generic_profile": profile,
            "seed": seed,
            "fixed_grid": grid,
            "selected_architecture": list(estimator.model_.hidden_widths),
            "validation_selection_score": validation_score,
            "test_metrics": metrics,
            "fit_seconds": fit_seconds,
            "history_entries": len(estimator.history_),
            "status": "completed",
            "run_signature": self.run_signature,
        }

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from importlib.metadata import PackageNotFoundError, version

            package_versions = {}
            for package in ("simple-adp", "openml", "scikit-learn", "torch", "numpy", "pandas"):
                try:
                    package_versions[package] = version(package)
                except PackageNotFoundError:
                    package_versions[package] = None
        except Exception:
            package_versions = {}
        run_config = {
            "run_signature": self.run_signature,
            "protocol_version": 2,
            "suite_keys": self.suites,
            "openml_suite_ids": {key: OPENML_SUITES[key].openml_id for key in self.suites},
            "profiles": self.profiles,
            "seeds": self.seeds,
            "search_settings": self.settings.as_dict(),
            "validation_fraction": self.validation_fraction,
            "screen_widths": self.screen_widths,
            "screen_depths": self.screen_depths,
            "deeper_screen_widths": self.deeper_screen_widths,
            "classification_ceiling": self.classification_ceiling,
            "regression_ceiling_r2": self.regression_ceiling_r2,
            "min_complexity_gain": self.min_complexity_gain,
            "min_hard_fold_fraction": self.min_hard_fold_fraction,
            "screen_seed_count": self.screen_seed_count,
            "package_versions": package_versions,
        }
        config_path = self.output_dir / f"run-config-{self.run_signature[:12]}.json"
        if not config_path.exists():
            config_path.write_text(
                json.dumps(run_config, indent=2, sort_keys=True, default=_json_default) + "\n",
                encoding="utf-8",
            )
        (self.output_dir / "active-run.json").write_text(
            json.dumps({"run_signature": self.run_signature, "config_file": config_path.name}, indent=2) + "\n",
            encoding="utf-8",
        )
        result_path = self.output_dir / "results.jsonl"
        screen_path = self.output_dir / "screening.jsonl"
        manifest_path = self.output_dir / "tasks.jsonl"
        completed: set[tuple[Any, ...]] = set()
        if result_path.exists():
            with result_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        if record.get("status") == "completed":
                            completed.add((record.get("run_signature"), record.get("dataset_sha256"), record["suite"], record["task_id"], record["repeat"], record["fold"], record["sample"], record["model"], record["seed"]))
                    except (ValueError, KeyError):
                        continue
        cached_screens: dict[tuple[Any, ...], dict[str, Any]] = {}
        screen_path = self.output_dir / "screening.jsonl"
        if screen_path.exists():
            with screen_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        if record.get("run_signature") == self.run_signature and record.get("decision"):
                            key = (record["suite"], record["task_id"], record["repeat"], record["fold"], record["sample"])
                            cached_screens[key] = record
                    except (ValueError, KeyError):
                        continue
        for suite_key in self.suites:
            task_ids = load_suite_tasks(suite_key)
            if self.task_limit is not None:
                task_ids = task_ids[:self.task_limit]
            for task_id in task_ids:
                try:
                    data = load_openml_task(suite_key, task_id)
                    dimensions = tuple(int(value) for value in data.task.get_split_dimensions())
                    task_manifest = self._task_manifest(data, dimensions)
                    task_manifest["run_signature"] = self.run_signature
                    _append_jsonl(manifest_path, task_manifest)
                    n_repeats, n_folds, n_samples = dimensions
                    fold_specs = [
                        (repeat, fold, sample)
                        for repeat in range(n_repeats)
                        for fold in range(n_folds)
                        for sample in range(n_samples)
                    ]
                    if self.fold_limit is not None:
                        fold_specs = fold_specs[:self.fold_limit]
                    expected_fold_count = len(fold_specs)
                    fold_work: list[tuple[FoldSplit, dict[str, Any]]] = []
                    for repeat, fold, sample in fold_specs:
                        split_seed = self.settings.seed + repeat * 100_003 + fold * 1_009 + sample
                        try:
                            split = make_fold_split(
                                data.task, data.y, data.suite, repeat=repeat, fold=fold, sample=sample,
                                validation_fraction=self.validation_fraction, split_seed=split_seed,
                            )
                            split_file = self._save_split(data, split)
                            screen_key = (suite_key, task_id, repeat, fold, sample)
                            screen_record = cached_screens.get(screen_key)
                            if (
                                screen_record is None
                                or screen_record.get("split_sha256") != split.split_sha256
                                or screen_record.get("dataset_sha256") != data.content_sha256
                            ):
                                screen = self._screen_fold(data, split)
                                screen_record = {
                                    "suite": suite_key, "task_id": task_id, "dataset_id": data.dataset_id,
                                    "dataset_sha256": data.content_sha256,
                                    "repeat": repeat, "fold": fold, "sample": sample,
                                    "split_sha256": split.split_sha256, "split_file": split_file,
                                    "run_signature": self.run_signature,
                                    **screen,
                                    "decision": (
                                        "exclude_fold_simple_mlp_saturated" if screen["saturated"] else
                                        "include_fold_for_full_comparison" if screen["complexity_sensitive"] else
                                        "exclude_fold_no_measured_complexity_gain"
                                    ),
                                }
                                _append_jsonl(screen_path, screen_record)
                                cached_screens[screen_key] = screen_record
                            fold_work.append((split, screen_record))
                        except Exception as exc:
                            _append_jsonl(screen_path, {
                                "suite": suite_key, "task_id": task_id, "repeat": repeat, "fold": fold,
                                "sample": sample, "status": "screen_failed", "run_signature": self.run_signature,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                    eligible_folds = [
                        item for item in fold_work
                        if item[1].get("decision") == "include_fold_for_full_comparison"
                    ]
                    hard_fraction = len(eligible_folds) / expected_fold_count if expected_fold_count else 0.0
                    dataset_eligible = hard_fraction >= self.min_hard_fold_fraction
                    decision = {
                        "suite": suite_key,
                        "task_id": task_id,
                        "dataset_id": data.dataset_id,
                        "dataset_sha256": data.content_sha256,
                        "dataset_name": data.name,
                        "run_signature": self.run_signature,
                        "expected_folds": expected_fold_count,
                        "screened_folds": len(fold_work),
                        "complexity_sensitive_folds": len(eligible_folds),
                        "complexity_sensitive_fraction": hard_fraction,
                        "minimum_fraction": self.min_hard_fold_fraction,
                        "eligible_for_full_comparison": dataset_eligible,
                        "decision": "include_dataset" if dataset_eligible else "exclude_dataset_insufficient_hard_folds",
                    }
                    _append_jsonl(self.output_dir / "dataset-decisions.jsonl", decision)
                    if self.screen_only or not dataset_eligible:
                        continue
                    for split, screen_record in eligible_folds:
                        repeat, fold, sample = screen_record["repeat"], screen_record["fold"], screen_record["sample"]
                        for model_id, profile, grid in model_runs(self.profiles):
                            model_label = model_id if profile is None else f"{model_id}@{profile}"
                            for seed in self.seeds:
                                run_key = (self.run_signature, data.content_sha256, suite_key, task_id, repeat, fold, sample, model_label, seed)
                                if run_key in completed:
                                    continue
                                try:
                                    record = self._run_one(data, split, model_id, profile, grid, seed)
                                    record["run_signature"] = self.run_signature
                                    record["dataset_sha256"] = data.content_sha256
                                    _append_jsonl(result_path, record)
                                    completed.add(run_key)
                                except Exception as exc:
                                    _append_jsonl(result_path, {
                                        "suite": suite_key, "task_id": task_id, "dataset_id": data.dataset_id,
                                        "dataset_sha256": data.content_sha256,
                                        "dataset_name": data.name, "repeat": repeat, "fold": fold,
                                        "sample": sample, "split_sha256": split.split_sha256,
                                        "model": model_label, "model_id": model_id, "generic_profile": profile,
                                        "seed": seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                        "run_signature": self.run_signature,
                                    })
                except Exception as exc:
                    _append_jsonl(manifest_path, {
                        "suite": suite_key, "task_id": task_id, "status": "task_load_failed",
                        "run_signature": self.run_signature, "error": f"{type(exc).__name__}: {exc}",
                    })
