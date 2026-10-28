"""One-epoch execution smoke coverage for every public model and task adapter.

These tests intentionally use one full-batch optimizer step per architecture.
They check that APIs connect end to end; they make no quality claim.
"""

import numpy as np
import pytest
import torch

from simple_adp import (
    MODEL_REGISTRY,
    CallableTask,
    ClassificationTask,
    RegressionTask,
    SearchSettings,
    TabularADP,
    vanilla_mlp_grid,
)


SMOKE_ROUTES = tuple(
    (model_id, profile)
    for model_id, config in MODEL_REGISTRY.items()
    for profile in (("faithful", "corrected") if config.generic else ("faithful",))
)


@pytest.fixture(params=["classification", "regression"])
def task_data(request):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(12, 3)).astype(np.float32)
    if request.param == "classification":
        y = np.where(X[:, 0] + X[:, 1] > 0, "positive", "negative")
        adapter = ClassificationTask()
    else:
        y = (X[:, 0] * X[:, 1] - X[:, 2]).astype(np.float32)
        adapter = RegressionTask()
    return X[:8], y[:8], X[8:], y[8:], adapter, request.param


@pytest.mark.parametrize(("model_id", "profile"), SMOKE_ROUTES)
def test_every_registered_model_profile_one_epoch_full_batch(model_id, profile, task_data, tmp_path):
    X_train, y_train, X_valid, y_valid, adapter, task_name = task_data
    settings = SearchSettings(
        init_width=1,
        init_depth=1,
        max_epochs=1,
        patience_es=1,
        patience_width_exp=1,
        patience_depth_exp=1,
        max_width=2,
        max_depth=2,
        max_neurons=16,
        batch_size=len(X_train),
        max_candidates=1,
        seed=11,
    )
    fixed_grid = vanilla_mlp_grid() if model_id == "MLP-Fixed" else None

    estimator = TabularADP(
        model_id,
        task=adapter,
        settings=settings,
        generic_profile=profile,
        fixed_grid=fixed_grid,
    )
    estimator.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
    )
    predictions = estimator.predict(X_valid)
    assert predictions.shape[0] == len(X_valid)
    assert estimator.model_ is not None
    assert estimator.history_

    if model_id == "ADP-W-All" and task_name == "classification":
        probabilities = estimator.predict_proba(X_valid)
        assert probabilities.shape == (len(X_valid), 2)

    # Exercise the fitted-artifact save/load prediction path once per task.
    if model_id == "ADP-W-All":
        artifact = tmp_path / f"{task_name}.pt"
        estimator.save(artifact)
        restored = TabularADP.load(artifact)
        np.testing.assert_array_equal(predictions, restored.predict(X_valid))


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
def test_generic_profiles_and_custom_task_one_epoch_full_batch(profile):
    rng = np.random.default_rng(23)
    X = rng.normal(size=(10, 2)).astype(np.float32)
    y = X[:, 0] - 2 * X[:, 1]
    custom = CallableTask(
        1,
        lambda output, target: torch.nn.functional.l1_loss(output, target.reshape(-1, 1)),
        lambda output, target: float(torch.nn.functional.l1_loss(output, target.reshape(-1, 1))),
        lambda output: output[:, 0],
        task_name="smoke-mae",
    )
    estimator = TabularADP(
        "ADP-WD-RR-D1",
        task=custom,
        settings=SearchSettings(
            init_width=1,
            init_depth=1,
            max_epochs=1,
            patience_es=1,
            patience_width_exp=0,
            patience_depth_exp=0,
            max_width=2,
            max_depth=2,
            max_neurons=16,
            batch_size=8,
            max_candidates=1,
            seed=13,
        ),
        generic_profile=profile,
    )
    estimator.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))
    assert estimator.predict(X[8:]).shape == (2,)
