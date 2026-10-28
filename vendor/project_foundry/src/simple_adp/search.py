"""Search facade dispatching to dedicated DEN/LG-ND and generic ADP controllers."""

from __future__ import annotations

import copy
import random
from typing import Callable, Sequence

import numpy as np
import torch

from .config import GenericProfile, ModelConfig, SearchSettings
from .dedicated import DedicatedController
from .generic_search import GenericCandidateSearch, MutationFunction
from .models import TabularMLP
from .results import SearchResult
from .tasks import TaskAdapter
from .training import StageResult, StageTrainer


class SearchController:
    """Builds stage tensors/models and routes each registered model to its controller."""

    def __init__(
        self,
        task: TaskAdapter,
        settings: SearchSettings,
        *,
        generic_profile: GenericProfile = "faithful",
        checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
        generic_widen_fn: MutationFunction | None = None,
        generic_deepen_fn: MutationFunction | None = None,
        generic_mutation_id: str | None = None,
        generic_artifact_store=None,
    ) -> None:
        self.task = task
        self.settings = settings
        self.generic_profile = generic_profile
        self.checkpoint_callback = checkpoint_callback
        self.generic_widen_fn = generic_widen_fn
        self.generic_deepen_fn = generic_deepen_fn
        self.generic_mutation_id = generic_mutation_id
        self.generic_artifact_store = generic_artifact_store

    @staticmethod
    def _validate_resume_payload(config: ModelConfig, payload: dict[str, object]) -> None:
        """Reject malformed controller state as an unrecoverable resume error."""
        def fail(message: str, cause: Exception | None = None) -> None:
            error = ValueError(f"exact-resume failure: {message}")
            if cause is None:
                raise error
            raise error from cause

        if not isinstance(payload, dict):
            fail("checkpoint payload is not a mapping")
        if payload.get("search_status") == "completed":
            snapshot = payload.get("global_best_model_snapshot")
            if not isinstance(snapshot, dict):
                fail("completed checkpoint lacks global-best model snapshot")
            try:
                TabularMLP.from_snapshot(snapshot)
                float(payload["global_best_score"])
            except Exception as exc:
                fail(f"completed checkpoint best model cannot be restored: {exc}", exc)
            return

        context = payload.get("search_context")
        stage = payload.get("stage_state")
        if not isinstance(context, dict) or not isinstance(stage, dict):
            fail("checkpoint lacks resumable controller and stage state")
        if context.get("controller") != config.id:
            fail("checkpoint controller identity does not match the selected model")
        phase = context.get("phase")
        if config.controller == "phase_search":
            phases = {"initial", "width", "depth"}
        elif config.controller == "depth_den":
            phases = {"initial", "depth", "progressive_depth_step"}
        elif config.controller == "fixed":
            phases = {"fixed"}
        else:
            phases = {"initial", "width"}
        if not isinstance(phase, str) or phase not in phases:
            fail(f"unsupported {config.controller} controller phase {phase!r}")

        try:
            current_widths = [int(width) for width in context["current_architecture"]]
            stage_widths = [int(width) for width in stage["hidden_widths"]]
            if not current_widths or any(width <= 0 for width in current_widths):
                fail("checkpoint controller architecture is invalid")
            if current_widths != stage_widths:
                fail("checkpoint controller and stage architectures do not match")
            history = context.get("history", [])
            if not isinstance(history, list):
                fail("checkpoint controller history is not a list")
            counters = context.get("counters", {})
            if not isinstance(counters, dict):
                fail("checkpoint controller counters are not a mapping")
            for value in counters.values():
                int(value)
            for key in (
                "fixed_index", "growth_cursor", "candidate_count", "current_failures",
                "phase_index", "progressive_start_width",
            ):
                if key in context:
                    int(context[key])
            for key in ("global_best_score", "current_score", "previous_candidate_score"):
                if key in context:
                    float(context[key])
        except Exception as exc:
            if "exact-resume failure:" in str(exc):
                raise
            fail(f"checkpoint controller state is malformed: {exc}", exc)

        needs_best = phase not in {"initial"} and not (
            config.controller == "fixed" and int(context.get("fixed_index", 0)) == 0
        )
        if needs_best:
            snapshot = context.get("global_best_snapshot")
            if not isinstance(snapshot, dict):
                fail("checkpoint controller is missing its global-best snapshot")
            try:
                TabularMLP.from_snapshot(snapshot)
                float(context["global_best_score"])
            except Exception as exc:
                fail(f"checkpoint global-best model cannot be restored: {exc}", exc)

    def _stage(
        self,
        model: TabularMLP,
        arrays: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        generic: bool = False,
        resume_state: dict[str, object] | None = None,
        search_context: dict[str, object] | None = None,
    ) -> StageResult:
        callback = None
        if self.checkpoint_callback is not None:
            context = copy.deepcopy(search_context or {})

            def callback(stage_state: dict[str, object]) -> None:
                record = copy.deepcopy(stage_state)
                record["search_context"] = context
                self.checkpoint_callback(record)

        trainer = StageTrainer(
            self.task,
            self.settings,
            optimizer_kind="adamw" if generic else "adam",
            use_penalties=generic,
            checkpoint_callback=callback,
        )
        return trainer.train_stage(model, *arrays, resume_state=resume_state)

    @staticmethod
    def _arrays(x_train, y_train, x_valid, y_valid, task, device):
        return (
            torch.as_tensor(x_train, dtype=torch.float32, device=device),
            task.encode_targets(y_train, device=torch.device(device)),
            torch.as_tensor(x_valid, dtype=torch.float32, device=device),
            task.encode_targets(y_valid, device=torch.device(device)),
        )

    def _initial_model(self, input_dim: int, output_dim: int) -> TabularMLP:
        random.seed(self.settings.seed)
        np.random.seed(self.settings.seed)
        torch.manual_seed(self.settings.seed)
        return TabularMLP(
            input_dim,
            [self.settings.init_width] * self.settings.init_depth,
            output_dim,
        ).to(self.settings.device)

    def run(
        self,
        config: ModelConfig,
        x_train,
        y_train,
        x_valid,
        y_valid,
        *,
        input_dim: int,
        output_dim: int,
        fixed_grid: Sequence[Sequence[int]] | None = None,
        resume_payload: dict[str, object] | None = None,
        artifact_resume_context: dict[str, object] | None = None,
    ) -> SearchResult:
        arrays = self._arrays(x_train, y_train, x_valid, y_valid, self.task, self.settings.device)
        if resume_payload is not None:
            self._validate_resume_payload(config, resume_payload)
        if artifact_resume_context is not None and not config.generic:
            raise ValueError("candidate artifact recovery is only supported for generic search")
        if resume_payload is not None and resume_payload.get("search_status") == "completed":
            snapshot = resume_payload.get("global_best_model_snapshot")
            model = TabularMLP.from_snapshot(snapshot).to(self.settings.device)
            history = resume_payload.get("search_history", [])
            state = resume_payload.get("search_state", {})
            return SearchResult(model, float(resume_payload["global_best_score"]), list(history), dict(state))
        initial = self._initial_model(input_dim, output_dim)
        if resume_payload is not None:
            context = resume_payload.get("search_context")
            stage = resume_payload.get("stage_state")
            if not isinstance(context, dict) or not isinstance(stage, dict):
                raise ValueError("exact-resume failure: checkpoint lacks resumable controller and stage state")
            widths = context.get("current_architecture", stage.get("hidden_widths"))
            if not isinstance(widths, (list, tuple)) or not widths:
                raise ValueError("exact-resume failure: checkpoint lacks current architecture for exact stage resume")
            initial = TabularMLP(input_dim, [int(width) for width in widths], output_dim).to(self.settings.device)
            setattr(initial, "_width_cursor", int(context.get("growth_cursor", stage.get("current_cursor", 0))))
        elif artifact_resume_context is not None:
            widths = artifact_resume_context.get("current_architecture")
            if not isinstance(widths, (list, tuple)) or not widths:
                raise ValueError("exact-resume failure: candidate artifact lacks current architecture")
            initial = TabularMLP(input_dim, [int(width) for width in widths], output_dim).to(self.settings.device)
            setattr(initial, "_width_cursor", int(artifact_resume_context.get("growth_cursor", 0)))
        stage_runner = self._stage
        if config.controller == "phase_search":
            return GenericCandidateSearch(
                self.settings, self.generic_profile, stage_runner,
                widen_fn=self.generic_widen_fn,
                deepen_fn=self.generic_deepen_fn,
                mutation_id=self.generic_mutation_id,
                artifact_store=self.generic_artifact_store,
            ).run(
                initial,
                arrays,
                config,
                resume_payload=resume_payload,
                artifact_resume_context=artifact_resume_context,
            )
        return DedicatedController(self.settings, stage_runner).run(
            initial, arrays, config, fixed_grid=fixed_grid, resume_payload=resume_payload
        )

    def run_nested_width_depth(
        self,
        initial: TabularMLP,
        arrays,
        *,
        seed_new_depth_to_current_width: bool = False,
        width_growth_strategy: str = "all_layers",
        resume_payload: dict[str, object] | None = None,
    ) -> SearchResult:
        """Run the guide's separate nested controller; it is not a registry ID."""
        state_dict = getattr(self.task, "state_dict", None)
        task_state = state_dict() if callable(state_dict) else {"type": type(self.task).__qualname__}
        return DedicatedController(self.settings, self._stage).run_nested_width_depth(
            initial, arrays, resume_payload=resume_payload,
            task_state=task_state,
            seed_new_depth_to_current_width=seed_new_depth_to_current_width,
            width_growth_strategy=width_growth_strategy,
        )
