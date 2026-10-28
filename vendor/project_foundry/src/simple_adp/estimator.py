"""Public estimator that composes preprocessing, task adapters, and ADP search."""

from __future__ import annotations

import os
import tempfile
import hashlib
import copy
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .checkpoint import CheckpointStore, stable_signature
from .baselines import vanilla_mlp_grid
from .config import GenericProfile, SearchSettings
from .models import TabularMLP
from .preprocessing import TabularPreprocessor
from .registry import get_model_config
from .search import SearchController
from .tasks import ClassificationTask, RegressionTask, TaskAdapter
from .generic_search import MutationFunction
from .generic_artifacts import GenericSearchArtifactStore


class TabularADP:
    """Shared tabular estimator; validation data is explicit and fixed during search."""

    def __init__(
        self,
        model_id: str = "ADP-W-All",
        *,
        task: TaskAdapter | None = None,
        settings: SearchSettings | None = None,
        generic_profile: GenericProfile = "faithful",
        feature_schema=None,
        fixed_grid: Sequence[Sequence[int]] | None = None,
        generic_widen_fn: MutationFunction | None = None,
        generic_deepen_fn: MutationFunction | None = None,
        generic_mutation_id: str | None = None,
    ) -> None:
        self.config = get_model_config(model_id)
        self.task = task or RegressionTask()
        self.settings = settings if settings is not None else SearchSettings(
            patience_width_exp=self.config.patience_width_exp,
            patience_depth_exp=self.config.patience_depth_exp,
            delta_width=self.config.delta_width,
            delta_depth=self.config.delta_depth,
            max_width=self.config.max_width,
            max_depth=self.config.max_depth,
            max_neurons=self.config.max_neurons,
        )
        if generic_profile not in {"faithful", "corrected"}:
            raise ValueError("generic_profile must be faithful or corrected")
        self.generic_profile = generic_profile
        self.feature_schema = feature_schema
        self.fixed_grid = fixed_grid
        if (generic_widen_fn is not None or generic_deepen_fn is not None) and not generic_mutation_id:
            raise ValueError("custom generic mutation functions require a stable generic_mutation_id")
        self.generic_widen_fn = generic_widen_fn
        self.generic_deepen_fn = generic_deepen_fn
        self.generic_mutation_id = generic_mutation_id or "builtin"
        self.preprocessor_: TabularPreprocessor | None = None
        self.model_: TabularMLP | None = None
        self.history_: list[dict[str, object]] = []
        self.search_state_: dict[str, object] = {}

    @staticmethod
    def _validation(validation_data, X_valid, y_valid):
        if validation_data is not None:
            if X_valid is not None or y_valid is not None or len(validation_data) != 2:
                raise ValueError("provide validation_data=(X, y) or X_valid and y_valid, not both")
            return validation_data
        if X_valid is None or y_valid is None:
            raise ValueError("explicit validation data is required")
        return X_valid, y_valid

    def fit(
        self,
        X,
        y,
        *,
        validation_data=None,
        X_valid=None,
        y_valid=None,
        fixed_grid: Sequence[Sequence[int]] | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        candidate_artifact_dir: str | os.PathLike[str] | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> "TabularADP":
        X_valid, y_valid = self._validation(validation_data, X_valid, y_valid)
        self.task.fit_targets(y)
        preprocessor = TabularPreprocessor(self.feature_schema)
        x_train = preprocessor.fit_transform(X)
        x_validation = preprocessor.transform(X_valid)
        # Keep raw task labels here; SearchController owns target encoding for stages.
        y_train = y
        y_validation = y_valid
        grid = fixed_grid if fixed_grid is not None else self.fixed_grid
        if self.config.controller == "fixed" and not grid:
            raise ValueError("MLP-Fixed requires an explicitly supplied fixed_grid")
        if (self.config.generic and self.generic_mutation_id != "builtin"
                and self.generic_widen_fn is None and self.generic_deepen_fn is None):
            raise ValueError("fitting a custom-mutation checkpoint requires its mutation functions")
        data_hasher = hashlib.sha256()
        for array in (x_train, y_train, x_validation, y_validation):
            contiguous = np.ascontiguousarray(array)
            data_hasher.update(str(contiguous.dtype).encode())
            data_hasher.update(str(contiguous.shape).encode())
            if contiguous.dtype.hasobject:
                import json
                data_hasher.update(json.dumps(contiguous.tolist(), default=repr, separators=(",", ":")).encode())
            else:
                data_hasher.update(contiguous.tobytes())
        signature = stable_signature({
            "config": self.config.signature_dict(),
            "settings": self.settings.as_dict(),
            "profile": self.generic_profile,
            "generic_mutation_id": self.generic_mutation_id,
            "input_dim": preprocessor.output_dim,
            "output_dim": self.task.output_dim,
            "feature_schema": self.feature_schema,
            "fitted_feature_columns": [(column.name, column.kind) for column in preprocessor.columns_ or ()],
            "fitted_feature_outputs": preprocessor.output_names_,
            "fixed_grid": [list(row) for row in grid] if grid is not None else None,
            "task": self.task.state_dict(),
            "data_sha256": data_hasher.hexdigest(),
        })
        store = CheckpointStore(checkpoint_path) if checkpoint_path else None
        # The per-batch estimator checkpoint is authoritative whenever present;
        # validate its identity before opening the replay-artifact hierarchy.
        resume_payload = (
            store.load(signature=signature, map_location=self.settings.device)
            if store is not None and store.path.exists() else None
        )
        if candidate_artifact_dir is not None and not self.config.generic:
            raise ValueError("candidate_artifact_dir is supported only for generic ADP search")
        artifact_root = None
        if candidate_artifact_dir is not None:
            artifact_root = Path(candidate_artifact_dir)
        elif self.config.generic and checkpoint_path is not None:
            checkpoint_file = Path(checkpoint_path)
            artifact_root = checkpoint_file.with_name(checkpoint_file.name + ".search")
        artifact_store = (
            GenericSearchArtifactStore(artifact_root, signature=signature)
            if artifact_root is not None and self.config.generic else None
        )
        artifact_recovery = (
            artifact_store.load_resume()
            if artifact_store is not None and resume_payload is None
            else None
        )
        if artifact_recovery is not None and artifact_recovery.get("completed"):
            state = artifact_recovery["state"]
            model = TabularMLP.from_snapshot(artifact_recovery["best_snapshot"]).to(self.settings.device)
            self.preprocessor_ = preprocessor
            self.model_ = model
            self.history_ = list(state.get("history", []))
            self.search_state_ = dict(state.get("search_state", {}))
            return self
        artifact_resume_context = (
            artifact_recovery.get("context")
            if artifact_recovery is not None and not artifact_recovery.get("completed") else None
        )
        last_checkpoint_payload: dict[str, object] = {}

        def persist(stage_state: dict[str, object]) -> None:
            context = stage_state.get("search_context", {})
            context = context if isinstance(context, dict) else {}
            if store is not None:
                stage_architecture = stage_state.get("hidden_widths")
                counters = dict(context.get("counters", {}))
                counters.update({
                    key: stage_state.get(key)
                    for key in ("epoch", "batch_cursor", "batches_done", "patience")
                })
                record = {
                    "configuration_id": self.config.id,
                    "model_configuration": self.config.signature_dict(),
                    "preprocessor": preprocessor.state_dict(),
                    "task": self.task.state_dict(),
                    "current_architecture": context.get("current_architecture", stage_architecture),
                    "best_architecture": context.get("global_best_architecture", stage_architecture),
                    "growth_cursor": context.get("growth_cursor", stage_state.get("best_cursor", 0)),
                    "global_best_score": context.get("global_best_score", stage_state.get("best_score")),
                    "counters": counters,
                    "search_context": context,
                    "global_best_model_snapshot": context.get("global_best_snapshot"),
                    "optimizer_state": stage_state.get("optimizer_state"),
                    "scheduler_state": stage_state.get("scheduler_state"),
                    "rng_state": stage_state.get("rng_state"),
                    "stage_state": stage_state,
                    "data_position": {"epoch": stage_state.get("epoch"), "batch_cursor": stage_state.get("batch_cursor"), "permutation": stage_state.get("permutation")},
                }
                last_checkpoint_payload.clear()
                last_checkpoint_payload.update(copy.deepcopy(record))
                store.save(record, signature=signature)
                mirror_checkpoint = getattr(progress_callback, "persist_training_checkpoint", None)
                if callable(mirror_checkpoint):
                    mirror_checkpoint(record)
            if progress_callback is not None:
                progress_callback({
                    "stage_status": stage_state.get("stage_status", "active"),
                    "hidden_widths": stage_state.get("hidden_widths"),
                    "current_cursor": stage_state.get("current_cursor", 0),
                    "epoch": stage_state.get("epoch", 0),
                    "batch_cursor": stage_state.get("batch_cursor", 0),
                    "batches_done": stage_state.get("batches_done", 0),
                    "patience": stage_state.get("patience"),
                    "best_score": stage_state.get("best_score"),
                    "search_context": context,
                    "rng_state": stage_state.get("rng_state"),
                })

        controller = SearchController(
            self.task,
            self.settings,
            generic_profile=self.generic_profile,
            checkpoint_callback=persist if store is not None or progress_callback is not None else None,
            generic_widen_fn=self.generic_widen_fn,
            generic_deepen_fn=self.generic_deepen_fn,
            generic_mutation_id=self.generic_mutation_id,
            generic_artifact_store=artifact_store,
        )
        if resume_payload is not None:
            # Signature mismatch is fatal: never continue a checkpoint against
            # different data, task, preprocessing, model, profile, or settings.
            reconcile = getattr(progress_callback, "reconcile_checkpoint", None)
            if callable(reconcile):
                reconcile(resume_payload)
            last_checkpoint_payload.update(copy.deepcopy(resume_payload))
        result = controller.run(
            self.config,
            x_train,
            y_train,
            x_validation,
            y_validation,
            input_dim=preprocessor.output_dim,
            output_dim=self.task.output_dim,
            fixed_grid=grid,
            resume_payload=resume_payload,
            artifact_resume_context=artifact_resume_context,
        )
        if store is not None:
            current_architecture = result.state.get("current_architecture", result.state.get("current_widths", result.model.hidden_widths))
            final_record = copy.deepcopy(last_checkpoint_payload)
            final_record.update({
                "configuration_id": self.config.id,
                "model_configuration": self.config.signature_dict(),
                "current_architecture": list(current_architecture),
                "best_architecture": list(result.model.hidden_widths),
                "growth_cursor": int(getattr(result.model, "_width_cursor", 0)),
                "global_best_score": result.score,
                "global_best_model_snapshot": result.model.snapshot(cursor=int(getattr(result.model, "_width_cursor", 0))),
                "search_state": result.state,
                "counters": {
                    "architecture_search": result.state,
                    "last_stage": final_record.get("counters", {}),
                },
                "search_history": result.history,
                "search_status": "completed",
            })
            store.save(final_record, signature=signature)
        self.preprocessor_ = preprocessor
        self.model_ = result.model.to(self.settings.device)
        self.history_ = result.history
        self.search_state_ = result.state
        return self

    def _outputs(self, X) -> torch.Tensor:
        if self.model_ is None or self.preprocessor_ is None:
            raise RuntimeError("fit must be called before prediction")
        features = torch.as_tensor(self.preprocessor_.transform(X), dtype=torch.float32, device=self.settings.device)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(features)

    def predict(self, X) -> np.ndarray:
        return self.task.predictions(self._outputs(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.task.predict_proba(self._outputs(X))

    def decision_function(self, X) -> np.ndarray:
        return self._outputs(X).detach().cpu().numpy()

    def save(self, path: str | os.PathLike[str]) -> None:
        if self.model_ is None or self.preprocessor_ is None:
            raise RuntimeError("cannot save an unfitted estimator")
        payload = {
            "format_version": 2,
            "model_id": self.config.id,
            "model_configuration": self.config.signature_dict(),
            "settings": self.settings.as_dict(),
            "generic_profile": self.generic_profile,
            "generic_mutation_id": self.generic_mutation_id,
            "feature_schema": self.feature_schema,
            "fixed_grid": self.fixed_grid,
            "task": self.task.state_dict(),
            "preprocessor": self.preprocessor_.state_dict(),
            "model": self.model_.snapshot(cursor=int(getattr(self.model_, "_width_cursor", 0))),
            "history": self.history_,
            "search_state": self.search_state_,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        task: TaskAdapter | None = None,
        map_location="cpu",
        generic_widen_fn: MutationFunction | None = None,
        generic_deepen_fn: MutationFunction | None = None,
    ) -> "TabularADP":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format_version") != 2:
            raise ValueError("unsupported estimator artifact version")
        task_state = payload["task"]
        if task is None:
            if task_state.get("type") == "RegressionTask":
                task = RegressionTask()
            elif task_state.get("type") == "ClassificationTask":
                task = ClassificationTask()
            else:
                raise ValueError("loading a custom TaskAdapter requires passing a compatible task instance")
        task.load_state_dict(task_state)
        settings = SearchSettings(**payload["settings"])
        estimator = cls(
            payload["model_id"], task=task, settings=settings,
            generic_profile=payload["generic_profile"],
            feature_schema=payload["feature_schema"], fixed_grid=payload["fixed_grid"],
            generic_widen_fn=generic_widen_fn,
            generic_deepen_fn=generic_deepen_fn,
            generic_mutation_id=payload.get("generic_mutation_id", "builtin"),
        )
        if payload.get("model_configuration") != estimator.config.signature_dict():
            raise ValueError("estimator artifact model configuration does not match the registered mechanism")
        preprocessor = TabularPreprocessor(estimator.feature_schema)
        preprocessor.load_state_dict(payload["preprocessor"])
        estimator.preprocessor_ = preprocessor
        estimator.model_ = TabularMLP.from_snapshot(payload["model"]).to(map_location)
        estimator.history_ = payload["history"]
        estimator.search_state_ = payload["search_state"]
        return estimator

