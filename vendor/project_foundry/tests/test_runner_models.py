import numpy as np
import pytest

from simple_adp import ClassificationTask, MODEL_REGISTRY, RegressionTask, SearchSettings
from simple_adp.baselines import vanilla_mlp_grid
from simple_adp.runner.models import SimpleADPModelAdapter
from simple_adp.runner.plan import SplitSpec
from simple_adp.runner.route import ModelRoute


ADP_ADAPTER_ROUTES = tuple(
    (model_id, profile)
    for model_id, config in MODEL_REGISTRY.items()
    if model_id != "MLP-Fixed"
    for profile in (("faithful", "corrected") if config.generic else (None,))
)


@pytest.mark.parametrize("task_kind", ["classification", "regression"])
@pytest.mark.parametrize("provider_data", ["array", "dataframe"])
def test_model_adapter_runs_tabular_dataset_provider(task_kind, provider_data):
    rng = np.random.default_rng(731)
    features = rng.normal(size=(12, 3)).astype(np.float32)
    if task_kind == "classification":
        target = np.where(features[:, 0] > 0, "positive", "negative")
    else:
        target = (features[:, 0] * features[:, 1] - features[:, 2]).astype(np.float32)
    if provider_data == "dataframe":
        import pandas as pd

        categories = np.asarray(["seen-a", "seen-b"] * 5 + ["unseen", "unseen"], dtype=object)
        X = pd.DataFrame(
            {"x0": features[:, 0], "x1": features[:, 1], "category": categories},
            index=np.arange(20, 32),
        )
    else:
        X = features
    data = type("ArrayTask", (), {
        "X": X,
        "y": target,
        "task_kind": task_kind,
        "feature_schema": None,
        "dataset_id": "array-task",
    })()
    split = SplitSpec(
        repeat=0,
        fold=0,
        sample=0,
        train_indices=tuple(range(8)),
        validation_indices=(8, 9),
        test_indices=(10, 11),
        split_sha256="synthetic-split",
    )
    route = ModelRoute("ADP-W-All", init_width=2, init_depth=1)
    settings = SearchSettings(
        init_width=2,
        init_depth=1,
        max_epochs=1,
        patience_es=1,
        patience_width_exp=0,
        patience_depth_exp=0,
        batch_size=8,
        max_width=2,
        max_depth=1,
        max_neurons=16,
        seed=29,
    )

    result = SimpleADPModelAdapter().fit_predict(
        route, data=data, split=split, seed=29, settings=settings,
    )

    assert result["selected_architecture"] == [2]
    assert result["history_entries"] >= 1
    assert result["test_metrics"]
    assert np.isfinite(result["validation_selection_score"])


def test_builtin_task_evaluation_metrics_have_explicit_values():
    classification = ClassificationTask()
    classification.fit_targets(np.asarray(["a", "b", "a", "b"]))
    class_metrics = classification.evaluation_metrics(
        np.asarray(["a", "b"]),
        np.asarray(["a", "b"]),
        np.asarray([[0.8, 0.2], [0.2, 0.8]]),
    )
    assert class_metrics["accuracy"] == 1.0
    assert class_metrics["balanced_accuracy"] == 1.0
    assert class_metrics["macro_f1"] == 1.0
    assert class_metrics["log_loss"] == pytest.approx(-np.log(0.8))

    regression = RegressionTask()
    regression.fit_targets(np.asarray([1.0, 2.0, 3.0]))
    regression_metrics = regression.evaluation_metrics(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 1.0, 5.0]),
    )
    assert regression_metrics["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert regression_metrics["mae"] == pytest.approx(1.0)
    assert regression_metrics["r2"] == pytest.approx(-1.5)


@pytest.mark.parametrize("task_kind", ["classification", "regression"])
@pytest.mark.parametrize(("model_id", "profile"), ADP_ADAPTER_ROUTES)
def test_runner_adapter_smokes_every_canonical_adp_route_full_batch(task_kind, model_id, profile):
    """Every canonical ADP identity/profile reaches fit and metric conversion."""
    rng = np.random.default_rng(733)
    X = rng.normal(size=(8, 2)).astype(np.float32)
    y = (
        np.asarray(["a", "b"] * 4)
        if task_kind == "classification"
        else (X[:, 0] * X[:, 1] - X[:, 1]).astype(np.float32)
    )
    data = type("ADPTask", (), {
        "X": X,
        "y": y,
        "task_kind": task_kind,
        "feature_schema": None,
        "dataset_id": "adp-route-smoke",
    })()
    split = SplitSpec(
        repeat=0,
        fold=0,
        sample=0,
        train_indices=(0, 1, 2, 3),
        validation_indices=(4, 5),
        test_indices=(6, 7),
        split_sha256=f"adp-{task_kind}",
    )
    route = ModelRoute(
        model_id,
        generic_profile=profile,
        label=f"{model_id}@{profile or 'dedicated'}",
        suite_group="width_only",
        init_width=1,
        init_depth=1,
    )
    settings = SearchSettings(
        init_width=1,
        init_depth=1,
        max_epochs=1,
        patience_es=1,
        patience_width_exp=1,
        patience_depth_exp=1,
        max_width=1,
        max_depth=1,
        max_neurons=16,
        batch_size=len(split.train_indices),
        max_candidates=1,
        seed=31,
    )

    result = SimpleADPModelAdapter().fit_predict(
        route, data=data, split=split, seed=31, settings=settings,
    )

    assert result["selected_architecture"] == [1]
    assert result["history_entries"] >= 1
    assert result["test_metrics"]


@pytest.mark.parametrize("task_kind", ["classification", "regression"])
@pytest.mark.parametrize("architecture", vanilla_mlp_grid())
def test_runner_adapter_smokes_each_vanilla_architecture_full_batch(task_kind, architecture):
    """Every individually planned fixed MLP route fits through the worker adapter."""
    rng = np.random.default_rng(809)
    X = rng.normal(size=(8, 2)).astype(np.float32)
    if task_kind == "classification":
        y = np.where(X[:, 0] + X[:, 1] >= 0, "positive", "negative")
    else:
        y = (X[:, 0] * X[:, 1] - X[:, 0]).astype(np.float32)
    data = type("VanillaTask", (), {
        "X": X,
        "y": y,
        "task_kind": task_kind,
        "feature_schema": None,
        "dataset_id": "vanilla-route-smoke",
    })()
    split = SplitSpec(
        repeat=0,
        fold=0,
        sample=0,
        train_indices=(0, 1, 2, 3),
        validation_indices=(4, 5),
        test_indices=(6, 7),
        split_sha256=f"vanilla-{task_kind}",
    )
    route = ModelRoute(
        "MLP-Fixed",
        fixed_grid=(tuple(architecture),),
        label=f"vanilla-d{len(architecture)}-w{architecture[0]}",
        suite_group="vanilla",
    )
    settings = SearchSettings(
        init_width=architecture[0],
        init_depth=len(architecture),
        max_epochs=1,
        patience_es=1,
        patience_width_exp=1,
        patience_depth_exp=1,
        max_width=512,
        max_depth=5,
        max_neurons=1_000_000_000,
        batch_size=len(split.train_indices),
        seed=37,
    )

    result = SimpleADPModelAdapter().fit_predict(
        route,
        data=data,
        split=split,
        seed=37,
        settings=settings,
    )

    assert result["selected_architecture"] == list(architecture)
    assert result["history_entries"] >= 1
    assert result["test_metrics"]


@pytest.mark.parametrize("task_kind", ["classification", "regression"])
def test_runner_adapter_smokes_aggregate_fixed_grid_full_batch(task_kind):
    """The aggregate vanilla route trains every supplied grid candidate."""
    rng = np.random.default_rng(811)
    X = rng.normal(size=(8, 2)).astype(np.float32)
    if task_kind == "classification":
        y = np.where(X[:, 0] + X[:, 1] >= 0, "positive", "negative")
    else:
        y = (X[:, 0] * X[:, 1] - X[:, 0]).astype(np.float32)
    data = type("AggregateVanillaTask", (), {
        "X": X,
        "y": y,
        "task_kind": task_kind,
        "feature_schema": None,
        "dataset_id": "aggregate-vanilla-route-smoke",
    })()
    split = SplitSpec(
        repeat=0,
        fold=0,
        sample=0,
        train_indices=(0, 1, 2, 3),
        validation_indices=(4, 5),
        test_indices=(6, 7),
        split_sha256=f"aggregate-vanilla-{task_kind}",
    )
    grid = tuple(tuple(widths) for widths in vanilla_mlp_grid())
    route = ModelRoute("MLP-Fixed", fixed_grid=grid, suite_group="vanilla")
    settings = SearchSettings(
        init_width=1,
        init_depth=1,
        max_epochs=1,
        patience_es=1,
        patience_width_exp=1,
        patience_depth_exp=1,
        max_width=512,
        max_depth=5,
        max_neurons=1_000_000_000,
        batch_size=len(split.train_indices),
        # The fixed-grid controller traverses its explicit grid independently
        # of the generic search candidate limit.
        max_candidates=1,
        seed=41,
    )

    result = SimpleADPModelAdapter().fit_predict(
        route, data=data, split=split, seed=41, settings=settings,
    )

    assert result["history_entries"] == len(grid) == 50
    assert tuple(result["selected_architecture"]) in grid
    assert result["test_metrics"]
