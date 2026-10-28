from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner import DatasetTaskRef, SplitSpec as PublicSplitSpec, TabularTaskData
from simple_adp.runner.datasets import OpenMLDatasetProvider
from simple_adp.runner.models import SimpleADPModelAdapter
from simple_adp.runner.plan import SplitSpec, SuitePlan


def test_builtin_openml_task_data_satisfies_typed_provider_payload_contract():
    from simple_adp import benchmark

    suite = benchmark.OPENML_SUITES["ctr23"]
    data = benchmark.TaskData(
        suite=suite,
        task_id=42,
        dataset_id=7,
        name="typed-regression-fixture",
        target_name="target",
        X=pd.DataFrame({"x": [0.0, 1.0]}),
        y=pd.Series([1.0, 2.0]),
        feature_schema=["numeric"],
        dataset_metadata={"source": "fixture"},
        content_sha256="stable-fixture-digest",
        task=object(),
    )

    assert isinstance(data, TabularTaskData)
    assert data.task_kind == "regression"
    first_adapter = data.task_adapter
    second_adapter = data.task_adapter
    assert type(first_adapter).__name__ == "RegressionTask"
    assert first_adapter is not second_adapter
    assert DatasetTaskRef("openml", "ctr23", "42").task_id == "42"
    assert PublicSplitSpec is SplitSpec


def test_split_archive_rejects_hash_that_does_not_match_indices(tmp_path):
    split = SplitSpec(0, 0, 0, (0, 1), (2,), (3,), "incorrect-digest")
    destination = tmp_path / "splits" / "split.npz"

    with pytest.raises(ValueError, match="signature does not match"):
        split.save(destination)

    assert not destination.exists()


@pytest.mark.parametrize("suite", ["cc18", "tabzilla", "ctr23"])
def test_openml_provider_plans_every_listed_task_for_registered_suite(
    suite, tmp_path, monkeypatch,
):
    """The OpenML adapter must pass every suite task through into the run plan."""
    from simple_adp import benchmark

    task_ids = [101, 203]
    monkeypatch.setattr(benchmark, "load_suite_tasks", lambda requested: task_ids)

    def load_task(requested_suite, task_id):
        assert requested_suite == suite
        return SimpleNamespace(
            X=pd.DataFrame({"x": np.arange(12, dtype=np.float32)}),
            y=pd.Series(np.arange(12, dtype=np.float32)),
            task=SimpleNamespace(get_split_dimensions=lambda: (1, 1, 1)),
            suite=benchmark.OPENML_SUITES[suite],
            task_id=task_id,
            dataset_id=task_id + 1000,
            name=f"dataset-{task_id}",
            target_name="target",
            content_sha256=f"content-{task_id}",
            feature_schema=["numeric"],
        )

    monkeypatch.setattr(benchmark, "load_openml_task", load_task)
    train_indices = np.arange(0, 8)
    validation_indices = np.arange(8, 10)
    test_indices = np.arange(10, 12)
    split_hash = benchmark._sha_indices(train_indices, validation_indices, test_indices)
    monkeypatch.setattr(
        benchmark,
        "make_fold_split",
        lambda *_args, **_kwargs: SimpleNamespace(
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            split_sha256=split_hash,
        ),
    )
    provider = OpenMLDatasetProvider()
    refs = provider.list_tasks(suite)
    assert [int(ref.task_id) for ref in refs] == task_ids
    assert all(ref.provider == "openml" and ref.suite == suite for ref in refs)

    settings = RunnerSettings(
        tasks=("width_only",),
        include_vanilla=False,
        run_hardness_screen=False,
        model_seeds=(7,),
        repeat_count=1,
        missing_present_task_repeats=1,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=tmp_path / "results",
        run_root=tmp_path / "run",
        dataset_providers=(provider,),
        dataset_suites=(("openml", suite),),
        model_adapters=(SimpleADPModelAdapter(),),
    )

    assert len(plan.runs) == 2 * 15  # two task IDs × three width-only models × five depths
    assert {run.task_ref.task_id for run in plan.runs} == {"101", "203"}
    assert {run.task_ref.dataset_id for run in plan.runs} == {"1101", "1203"}
    assert {run.route.suite_group for run in plan.runs} == {"width_only"}
    assert all(run.identity.split_sha256 == split_hash for run in plan.runs)
    for run in plan.runs:
        with np.load(run.split_file, allow_pickle=False) as saved_split:
            np.testing.assert_array_equal(saved_split["train"], np.arange(8))
            np.testing.assert_array_equal(saved_split["validation"], np.arange(8, 10))
            np.testing.assert_array_equal(saved_split["test"], np.arange(10, 12))
            assert saved_split["split_sha256"].item() == split_hash
        loaded = SplitSpec.load(
            Path(run.split_file), repeat=0, fold=0, sample=0,
        )
        assert loaded.split_sha256 == split_hash


@pytest.mark.parametrize("suite", ["cc18", "tabzilla", "ctr23"])
def test_openml_plan_crosses_every_task_with_every_registered_model_route(
    suite, tmp_path, monkeypatch,
):
    """Every listed task receives the complete model/profile/baseline route set."""
    import json

    from simple_adp import MODEL_REGISTRY, benchmark

    task_ids = [307, 509]
    monkeypatch.setattr(benchmark, "load_suite_tasks", lambda _suite: task_ids)

    def load_task(requested_suite, task_id):
        assert requested_suite == suite
        X = pd.DataFrame({
            "numeric": np.arange(8, dtype=np.float32) + task_id,
            "category": ["a", "b"] * 4,
        })
        return SimpleNamespace(
            X=X,
            y=pd.Series(np.asarray([0, 1] * 4)),
            task=SimpleNamespace(get_split_dimensions=lambda: (1, 1, 1)),
            suite=benchmark.OPENML_SUITES[suite],
            task_id=task_id,
            dataset_id=task_id + 2000,
            name=f"dataset-{task_id}",
            target_name="target",
            content_sha256=f"content-{task_id}",
            feature_schema=["numeric", "category"],
        )

    monkeypatch.setattr(benchmark, "load_openml_task", load_task)
    train = np.arange(0, 4)
    validation = np.arange(4, 6)
    test = np.arange(6, 8)
    split_hash = benchmark._sha_indices(train, validation, test)
    monkeypatch.setattr(
        benchmark,
        "make_fold_split",
        lambda *_args, **_kwargs: SimpleNamespace(
            train_indices=train,
            validation_indices=validation,
            test_indices=test,
            split_sha256=split_hash,
        ),
    )

    provider = OpenMLDatasetProvider()
    adapter = SimpleADPModelAdapter()
    settings = RunnerSettings(
        model_seeds=(13,),
        repeat_count=1,
        missing_present_task_repeats=1,
    )
    expected_routes = adapter.enumerate_routes(settings.generic_profiles, settings)
    assert len(expected_routes) == 146
    assert {route.model_id for route in expected_routes} == set(MODEL_REGISTRY)
    assert sum(route.screening_only for route in expected_routes) == 44

    plan = SuitePlan.build(
        settings=settings,
        results_dir=tmp_path / "results",
        run_root=tmp_path / "run",
        dataset_providers=(provider,),
        dataset_suites=(("openml", suite),),
        model_adapters=(adapter,),
    )

    assert len(plan.runs) == len(task_ids) * len(expected_routes)
    expected_route_records = sorted(
        json.dumps(route.as_dict(), sort_keys=True) for route in expected_routes
    )
    for task_id in (str(value) for value in task_ids):
        task_runs = [run for run in plan.runs if run.task_ref.task_id == task_id]
        assert len(task_runs) == len(expected_routes)
        assert sorted(json.dumps(run.route.as_dict(), sort_keys=True) for run in task_runs) == expected_route_records
        assert {run.identity.split_sha256 for run in task_runs} == {split_hash}
        assert all(run.identity.suite == suite for run in task_runs)
        for split_path in {run.split_file for run in task_runs}:
            with np.load(split_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["train"], train)
                np.testing.assert_array_equal(saved["validation"], validation)
                np.testing.assert_array_equal(saved["test"], test)
                assert saved["split_sha256"].item() == split_hash
