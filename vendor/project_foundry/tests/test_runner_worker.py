import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from simple_adp.runner import worker
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute, SimpleADPModelAdapter
from simple_adp.runner.plan import RunIdentity, SplitSpec


_WORKER_SMOKE_SETTINGS = RunnerSettings(smoke_test=True)
_WORKER_SMOKE_ROUTES = tuple(
    SimpleADPModelAdapter().enumerate_routes(
        _WORKER_SMOKE_SETTINGS.generic_profiles, _WORKER_SMOKE_SETTINGS,
    )
)


def test_worker_uses_coordinator_checkpoint_path_for_estimator_resume(tmp_path, monkeypatch):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="task", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="ADP-W-All", seed=17,
        split_sha256="split", run_signature="signature",
    )
    task_ref = DatasetTaskRef("toy", "suite", "task")
    route = ModelRoute("ADP-W-All", init_width=2, init_depth=1)
    result_dir = tmp_path / "results" / "job"
    spec_path = tmp_path / "run" / "spec.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(json.dumps({
        "identity": identity.as_dict(),
        "task_ref": {
            "provider": task_ref.provider,
            "suite": task_ref.suite,
            "task_id": task_ref.task_id,
            "dataset_id": task_ref.dataset_id,
            "metadata": task_ref.metadata,
        },
        "route": route.as_dict(),
        "model_adapter": "test-model",
        "split_file": str(tmp_path / "split.npz"),
        "result_dir": str(result_dir),
        "worker_settings": {
            "max_width": 8, "max_depth": 4, "max_neurons": 32,
            "batch_size": 8, "max_epochs_adp": 9, "smoke_test": True,
            "ex_k_width": 3, "ex_k_depth": 2,
            "patience_es": 7, "patience_width_exp": 4, "patience_depth_exp": 5,
            "delta_width": 0.125, "delta_depth": 0.25,
            "learning_rate": 0.002, "scheduler_t0": 6, "scheduler_t_mult": 3,
            "scheduler_eta_min": 0.0001, "weight_decay": 0.01, "grad_clip": 2.5,
            "lambda_loss": 0.75, "lambda_eq": 0.2, "lambda_ineq": 0.3,
            "max_candidates": 11, "min_new_layer_width": 4,
        },
    }))
    checkpoint_path = tmp_path / "resume" / "authoritative.checkpoint.pt"
    monkeypatch.setenv("SIMPLE_ADP_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("SIMPLE_ADP_JOB_ID", identity.run_id)
    startup_order = []
    monkeypatch.setattr(worker, "bootstrap_runtime", lambda *_args, **_kwargs: startup_order.append("bootstrap"))
    monkeypatch.setattr(worker, "enable_optional_gpu_accelerators", lambda: startup_order.append("backend"))
    monkeypatch.setattr(worker, "seed_everything", lambda _seed: startup_order.append("seed"))
    monkeypatch.setattr(worker, "resolve_torch_device", lambda: "cpu")
    monkeypatch.setattr(worker.SplitSpec, "load", lambda *_args, **_kwargs: SimpleNamespace(train_indices=(0, 1)))

    class ModelAdapter:
        def fit_predict(self, _route, **kwargs):
            assert kwargs["checkpoint_path"] == str(checkpoint_path)
            # Smoke mode overrides regular training settings at the worker
            # boundary: one epoch and one batch spanning all training rows.
            assert kwargs["settings"].max_epochs == 1
            assert kwargs["settings"].batch_size == 2
            for name, expected in {
                "ex_k_width": 3, "ex_k_depth": 2,
                "patience_es": 7, "patience_width_exp": 4, "patience_depth_exp": 5,
                "delta_width": 0.125, "delta_depth": 0.25,
                "learning_rate": 0.002, "scheduler_t0": 6, "scheduler_t_mult": 3,
                "scheduler_eta_min": 0.0001, "weight_decay": 0.01, "grad_clip": 2.5,
                "lambda_loss": 0.75, "lambda_eq": 0.2, "lambda_ineq": 0.3,
                "max_candidates": 11, "min_new_layer_width": 4,
            }.items():
                assert getattr(kwargs["settings"], name) == expected
            callback = kwargs["progress_callback"]
            assert callback is not None
            assert callback.checkpoint == str(checkpoint_path)
            return {
                "test_metrics": {}, "validation_metrics": {},
                "selected_architecture": [2], "fit_seconds": 0.1,
                "history_entries": 1, "artifact_path": None,
            }

    datasets = SimpleNamespace(get=lambda _name: SimpleNamespace(
        load_task=lambda _ref: SimpleNamespace(
            X=SimpleNamespace(), y=SimpleNamespace(), task_kind="custom",
            task_adapter=SimpleNamespace(), feature_schema=None,
        ),
    ))
    models = SimpleNamespace(get=lambda _name: ModelAdapter())
    monkeypatch.setattr(worker, "load_registries", lambda: startup_order.append("registry") or (datasets, models))

    worker.run_spec(spec_path)

    assert startup_order == ["bootstrap", "backend", "seed", "registry"]
    record = json.loads((result_dir / "result.json").read_text())
    assert record["status"] == "completed"


@pytest.mark.parametrize("task_kind", ["classification", "regression"])
@pytest.mark.parametrize(
    "route",
    _WORKER_SMOKE_ROUTES,
    ids=lambda route: route.route_label,
)
def test_worker_smokes_every_registered_route_with_one_epoch_full_training_batch(
    tmp_path, monkeypatch, task_kind, route,
):
    """Exercise route serialization and worker execution for the full registry."""
    X = np.asarray(
        [[index / 5.0, (index % 3) / 3.0] for index in range(10)],
        dtype=np.float32,
    )
    y = (
        np.asarray(["a", "b"] * 5)
        if task_kind == "classification"
        else (X[:, 0] * X[:, 1] - X[:, 1]).astype(np.float32)
    )
    data = SimpleNamespace(
        X=X,
        y=y,
        task_kind=task_kind,
        dataset_id=f"worker-smoke-{task_kind}",
        feature_schema=None,
        name=f"worker-smoke-{task_kind}",
        target_name="target",
        dataset_metadata={"source": "worker-smoke", "task_kind": task_kind},
    )
    train, validation, test = tuple(range(6)), (6, 7), (8, 9)
    digest = hashlib.sha256()
    for indices in (train, validation, test):
        digest.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
    split = SplitSpec(0, 0, 0, train, validation, test, digest.hexdigest())
    split_path = tmp_path / "split.npz"
    split.save(split_path)

    task_ref = DatasetTaskRef(
        "worker_smoke", task_kind, "task-0", data.dataset_id,
        {
            "dataset_name": data.name,
            "target_name": data.target_name,
            "dataset_metadata": data.dataset_metadata,
        },
    )
    identity = RunIdentity(
        provider=task_ref.provider,
        suite=task_ref.suite,
        task_id=task_ref.task_id,
        split_repeat=0,
        fold=0,
        sample=0,
        suite_repeat=1,
        model_route=route.route_label,
        seed=17,
        split_sha256=split.split_sha256,
        run_signature=f"worker-smoke-{task_kind}",
    )
    result_dir = tmp_path / "results" / identity.run_id
    spec_path = tmp_path / "run" / "run_spec.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(json.dumps({
        "identity": identity.as_dict(),
        "task_ref": {
            "provider": task_ref.provider,
            "suite": task_ref.suite,
            "task_id": task_ref.task_id,
            "dataset_id": task_ref.dataset_id,
            "metadata": task_ref.metadata,
        },
        "route": route.as_dict(),
        "model_adapter": "simple_adp",
        "split_file": str(split_path),
        "result_dir": str(result_dir),
        "worker_settings": {
            "max_epochs": 9,
            "max_epochs_adp": 9,
            "max_epochs_vanilla": 9,
            "batch_size": 2,
            "smoke_test": True,
            "init_width": 1,
            "init_depth": 1,
            "max_width": 512,
            "max_depth": 5,
            "max_neurons": 1_000_000_000,
            "patience_es": 1,
            "patience_width_exp": 0,
            "patience_depth_exp": 0,
            "max_candidates": 1,
            "seed": identity.seed,
        },
    }), encoding="utf-8")

    monkeypatch.setenv("SIMPLE_ADP_JOB_ID", identity.run_id)
    monkeypatch.delenv("SIMPLE_ADP_CHECKPOINT_PATH", raising=False)
    monkeypatch.setattr(worker, "bootstrap_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "enable_optional_gpu_accelerators", lambda: None)
    monkeypatch.setattr(worker, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(worker, "resolve_torch_device", lambda: "cpu")

    class Provider:
        def load_task(self, _ref):
            return data

    class RecordingAdapter:
        def fit_predict(self, requested_route, **kwargs):
            assert requested_route.route_label == route.route_label
            assert kwargs["settings"].max_epochs == 1
            assert kwargs["settings"].batch_size == len(train)
            return SimpleADPModelAdapter().fit_predict(requested_route, **kwargs)

    monkeypatch.setattr(
        worker, "load_registries",
        lambda: (
            SimpleNamespace(get=lambda _name: Provider()),
            SimpleNamespace(get=lambda _name: RecordingAdapter()),
        ),
    )

    record = worker.run_spec(spec_path)

    assert record["status"] == "completed"
    assert record["smoke_test"] is True
    assert record["model_id"] == route.model_id
    assert record["generic_profile"] == route.generic_profile
    assert record["history_entries"] >= 1
    assert record["split_sha256"] == split.split_sha256
    assert record["dataset_metadata"] == data.dataset_metadata
    assert np.isfinite(record["validation_selection_score"])
    if route.screening_only:
        assert record["test_metrics"] == {}
    else:
        assert record["test_metrics"]
