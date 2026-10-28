"""One child process executes exactly one registered dataset/model route."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import SearchSettings
from .artifacts import estimator_checkpoint_path
from .datasets import DatasetTaskRef
from .backend import enable_optional_gpu_accelerators, resolve_torch_device, seed_everything
from .route import ModelRoute
from .plan import RunIdentity, SplitSpec
from .runtime_tuning import bootstrap_runtime
from .state import atomic_write_json, utc_now
from .telemetry import telemetry_callback


def _rebuild_spec(value: dict[str, Any]) -> tuple[RunIdentity, DatasetTaskRef, ModelRoute]:
    identity_value = dict(value["identity"])
    identity_value.pop("run_id", None)
    identity = RunIdentity(**identity_value)
    ref = DatasetTaskRef(**value["task_ref"])
    route_value = dict(value["route"])
    route_value.pop("route_label", None)
    if route_value.get("fixed_grid") is not None:
        route_value["fixed_grid"] = tuple(tuple(int(item) for item in widths) for widths in route_value["fixed_grid"])
    route = ModelRoute(**route_value)
    return identity, ref, route


def load_registries():
    """Load model adapters only after worker backend initialization."""
    from .plugins import load_registries as load

    return load()


def run_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    identity, task_ref, route = _rebuild_spec(spec)
    if os.environ.get("SIMPLE_ADP_JOB_ID") not in {None, identity.run_id}:
        raise ValueError("worker job ID environment does not match the persisted run specification")
    bootstrap_runtime(f"suite-worker-{identity.run_id}", apply_os_tuning=True)
    enable_optional_gpu_accelerators()
    seed_everything(identity.seed)
    datasets, models = load_registries()
    provider = datasets.get(task_ref.provider)
    adapter = models.get(spec["model_adapter"])
    data = provider.load_task(task_ref)
    split = SplitSpec.load(
        Path(spec["split_file"]),
        repeat=identity.split_repeat,
        fold=identity.fold,
        sample=identity.sample,
    )
    settings_value = dict(spec["worker_settings"])
    smoke = bool(settings_value.get("smoke_test", False))
    device = resolve_torch_device()
    epoch_override = settings_value.get("max_epochs")
    if epoch_override is None:
        max_epochs = int(settings_value.get("max_epochs_vanilla" if route.model_id == "MLP-Fixed" else "max_epochs_adp", 0))
    else:
        max_epochs = int(epoch_override)
    if smoke:
        max_epochs = 1
    batch_size = int(settings_value.get("batch_size", 128))
    if smoke:
        batch_size = len(split.train_indices)
    search_settings = SearchSettings(
        init_width=int(settings_value.get("init_width", 16)),
        init_depth=int(settings_value.get("init_depth", 2)),
        max_epochs=max_epochs,
        batch_size=batch_size,
        max_width=int(settings_value["max_width"]),
        max_depth=int(settings_value["max_depth"]),
        max_neurons=int(settings_value["max_neurons"]),
        ex_k_width=int(settings_value.get("ex_k_width", 1)),
        ex_k_depth=int(settings_value.get("ex_k_depth", 1)),
        patience_es=int(settings_value.get("patience_es", 10)),
        patience_width_exp=int(settings_value.get("patience_width_exp", 10)),
        patience_depth_exp=int(settings_value.get("patience_depth_exp", 10)),
        delta_width=float(settings_value.get("delta_width", 0.0)),
        delta_depth=float(settings_value.get("delta_depth", 0.0)),
        learning_rate=float(settings_value.get("learning_rate", 1e-3)),
        scheduler_t0=int(settings_value.get("scheduler_t0", 10)),
        scheduler_t_mult=int(settings_value.get("scheduler_t_mult", 2)),
        scheduler_eta_min=float(settings_value.get("scheduler_eta_min", 0.0)),
        weight_decay=float(settings_value.get("weight_decay", 0.0)),
        grad_clip=(None if settings_value.get("grad_clip") is None else float(settings_value["grad_clip"])),
        lambda_loss=float(settings_value.get("lambda_loss", 1.0)),
        lambda_eq=float(settings_value.get("lambda_eq", 0.0)),
        lambda_ineq=float(settings_value.get("lambda_ineq", 0.0)),
        max_candidates=int(settings_value.get("max_candidates", 0)),
        min_new_layer_width=int(settings_value.get("min_new_layer_width", 1)),
        seed=identity.seed,
        device=device,
    )
    output_dir = Path(spec["result_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(
        os.environ.get("SIMPLE_ADP_CHECKPOINT_PATH")
        or estimator_checkpoint_path(output_dir)
    )
    started = time.perf_counter()
    try:
        result = adapter.fit_predict(
            route,
            data=data,
            split=split,
            seed=identity.seed,
            settings=search_settings,
            checkpoint_path=str(checkpoint_path),
            progress_callback=telemetry_callback(
                output_dir,
                job_id=identity.run_id,
                model_id=route.model_id,
                checkpoint_path=checkpoint_path,
                checkpoint_mirrors={
                    name: os.environ[key]
                    for name, key in {
                        "last": "SIMPLE_ADP_CHECKPOINT_LAST_PATH",
                        "best": "SIMPLE_ADP_CHECKPOINT_BEST_PATH",
                        "best_batch": "SIMPLE_ADP_CHECKPOINT_BEST_BATCH_PATH",
                    }.items()
                    if os.environ.get(key)
                },
            ),
        )
    except Exception as exc:
        record = {
            "job_id": identity.run_id,
            "run_signature": identity.run_signature,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "timestamp": utc_now(),
        }
        atomic_write_json(output_dir / "result.json", record)
        raise
    record = {
        "job_id": identity.run_id,
        "run_signature": identity.run_signature,
        "provider": task_ref.provider,
        "suite": task_ref.suite,
        "task_id": task_ref.task_id,
        "dataset_id": task_ref.dataset_id,
        "dataset_name": (task_ref.metadata or {}).get("dataset_name"),
        "target_name": (task_ref.metadata or {}).get("target_name"),
        "dataset_metadata": (task_ref.metadata or {}).get("dataset_metadata", {}),
        "split": {"repeat": identity.split_repeat, "fold": identity.fold, "sample": identity.sample},
        "repeat": identity.split_repeat,
        "fold": identity.fold,
        "sample": identity.sample,
        "split_sha256": identity.split_sha256,
        "model": route.route_label,
        "model_id": route.model_id,
        "generic_profile": route.generic_profile,
        "screening_only": route.screening_only,
        "screening_seed": route.screening_seed,
        "screening_architecture": list(route.screening_architecture) if route.screening_architecture else None,
        "task_kind": getattr(data, "task_kind", getattr(getattr(data, "suite", None), "task_kind", None)),
        "fixed_grid": [list(widths) for widths in route.fixed_grid] if route.fixed_grid else None,
        "seed": identity.seed,
        "status": "completed",
        "metrics": result.get("test_metrics", {}),
        "test_metrics": result.get("test_metrics", {}),
        "validation_metrics": result.get("validation_metrics", {}),
        "validation_selection_score": result.get("validation_selection_score"),
        "selected_architecture": result.get("selected_architecture"),
        "fit_seconds": result.get("fit_seconds"),
        "history_entries": result.get("history_entries"),
        "worker_seconds": time.perf_counter() - started,
        "artifact_path": result.get("artifact_path"),
        "smoke_test": smoke,
        "timestamp": utc_now(),
    }
    atomic_write_json(output_dir / "result.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one simple-adp suite route")
    parser.add_argument("--spec", required=True, help="path to a persisted RunSpec JSON file")
    args = parser.parse_args(argv)
    run_spec(args.spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
