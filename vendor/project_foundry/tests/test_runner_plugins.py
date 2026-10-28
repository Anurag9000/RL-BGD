import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import SplitSpec, SuitePlan
from simple_adp.runner.plugins import load_registries


class _EntryPoint:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def load(self):
        return self.value


class _EntryPoints:
    def __init__(self, entries):
        self.entries = entries

    def select(self, *, group):
        return [entry for entry_group, entry in self.entries if entry_group == group]


class _ExternalTask:
    def __init__(self):
        self.X = np.arange(24, dtype=np.float32).reshape(12, 2)
        self.y = np.asarray([0, 1] * 6)
        self.task_kind = "classification"
        self.dataset_id = "external-dataset"
        self.content_sha256 = "external-content"
        self.feature_schema = None
        self.name = "external-task"
        self.target_name = "label"
        self.dataset_metadata = {"source": "fixture", "version": 3}
        self.task_adapter = None


class _ExternalProvider:
    name = "external_provider"

    def list_tasks(self, suite):
        return [DatasetTaskRef(self.name, suite, "task-7")]

    def load_task(self, ref):
        assert ref.task_id == "task-7"
        return _ExternalTask()

    def list_splits(self, ref, data, settings):
        partitions = (tuple(range(8)), (8, 9), (10, 11))
        digest = hashlib.sha256()
        for indices in partitions:
            digest.update(np.asarray(indices, dtype=np.int64).tobytes())
        return [SplitSpec(0, 0, 0, *partitions, digest.hexdigest())]


class _ExternalModelAdapter:
    name = "external_model"

    def enumerate_routes(self, generic_profiles, settings=None):
        return [ModelRoute("CUSTOM-MLP", suite_group="width_only", init_width=3, init_depth=1)]

    def fit_predict(self, *args, **kwargs):
        raise AssertionError("planning must not execute model training")


def test_entry_point_plugins_extend_dataset_and_model_routes_without_coordinator_edits(
    monkeypatch, tmp_path,
):
    entries = _EntryPoints([
        ("simple_adp.dataset_providers", _EntryPoint("external", _ExternalProvider)),
        ("simple_adp.model_adapters", _EntryPoint("external", _ExternalModelAdapter)),
    ])
    monkeypatch.setattr("simple_adp.runner.plugins.entry_points", lambda: entries)

    datasets, models = load_registries()
    assert set(datasets.providers) == {"openml", "external_provider"}
    assert set(models.adapters) == {"simple_adp", "external_model"}

    settings = RunnerSettings(
        tasks=("width_only",),
        include_vanilla=False,
        run_hardness_screen=False,
        model_seeds=(11,),
        repeat_count=1,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=tmp_path / "results",
        run_root=tmp_path / "run",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("external_provider", "external_suite"),),
        model_adapters=tuple(models.adapters.values()),
    )

    custom_runs = [run for run in plan.runs if run.route.model_id == "CUSTOM-MLP"]
    assert len(custom_runs) == 1
    run = custom_runs[0]
    assert run.task_ref.key == "external_provider:external_suite:task-7"
    assert run.route.model_id == "CUSTOM-MLP"
    assert run.route.suite_group == "width_only"
    assert run.route.init_width == 3
    assert run.task_ref.metadata["dataset_metadata"] == {
        "source": "fixture", "version": 3,
    }
    assert run.identity.split_sha256 == hashlib.sha256(
        b"".join(np.asarray(indices, dtype=np.int64).tobytes() for indices in (
            tuple(range(8)), (8, 9), (10, 11),
        ))
    ).hexdigest()
    assert run.model_adapter == "external_model"


def test_dataset_provider_metadata_must_be_a_mapping(tmp_path):
    provider = _ExternalProvider()
    original_load = provider.load_task

    def load_with_invalid_metadata(ref):
        data = original_load(ref)
        data.dataset_metadata = ["not", "a", "mapping"]
        return data

    provider.load_task = load_with_invalid_metadata
    settings = RunnerSettings(
        tasks=("width_only",),
        include_vanilla=False,
        run_hardness_screen=False,
        model_seeds=(11,),
        repeat_count=1,
    )

    with pytest.raises(TypeError, match="dataset_metadata that is not a mapping"):
        SuitePlan.build(
            settings=settings,
            results_dir=tmp_path / "results",
            run_root=tmp_path / "run",
            dataset_providers=(provider,),
            dataset_suites=((provider.name, "external_suite"),),
            model_adapters=(_ExternalModelAdapter(),),
        )

