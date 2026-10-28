"""Candidate-based generic ADP search, kept separate from dedicated DEN policy."""

from __future__ import annotations

from typing import Any, Callable

from .config import GenericProfile, ModelConfig, SearchSettings
from .growth import (
    expand_depth, expand_depth_match_width, expand_width,
    expand_width_all_layers, total_neurons,
)
from .models import TabularMLP, from_snapshot_preserving_rng
from .results import SearchResult


MutationFunction = Callable[[TabularMLP, SearchSettings], TabularMLP | None]


class GenericCandidateSearch:
    """Implements adp_search's sequential candidate phases and compatibility profiles."""

    def __init__(
        self,
        settings: SearchSettings,
        profile: GenericProfile,
        stage_runner: Callable,
        *,
        widen_fn: MutationFunction | None = None,
        deepen_fn: MutationFunction | None = None,
        mutation_id: str | None = None,
        artifact_store: Any | None = None,
    ) -> None:
        self.settings = settings
        self.profile = profile
        self._stage_runner = stage_runner
        self.widen_fn = widen_fn
        self.deepen_fn = deepen_fn
        if (widen_fn is not None or deepen_fn is not None) and not mutation_id:
            raise ValueError("custom generic mutation functions require a stable mutation_id")
        self.mutation_id = mutation_id or "builtin"
        self.artifact_store = artifact_store
        self.context: dict[str, object] = {}

    def _commit_candidate(
        self,
        *,
        index: int,
        phase: str,
        model: TabularMLP,
        stage,
        score: float,
        accepted: bool,
        selected: TabularMLP,
        selected_score: float,
        current_score: float,
        previous_candidate_score: float,
        failures: int,
        failure_counts: dict[str, int],
        phase_index: int,
        history: list[dict[str, object]],
        candidate_count: int,
    ) -> None:
        cursor = int(getattr(model, "_width_cursor", 0))
        selected_cursor = int(getattr(selected, "_width_cursor", 0))
        self.context = {
            "controller": self.context.get("controller"),
            "phase": phase,
            "profile": self.profile,
            "mutation_id": self.mutation_id,
            "current_architecture": list(model.hidden_widths),
            "global_best_architecture": list(selected.hidden_widths),
            "global_best_score": selected_score,
            "global_best_snapshot": selected.snapshot(cursor=selected_cursor),
            "candidate_count": candidate_count,
            "counters": {f"{phase}_failures": failures},
            "current_failures": failures,
            "current_score": current_score,
            "previous_candidate_score": previous_candidate_score,
            "failure_counts": dict(failure_counts),
            "phase_index": phase_index,
            "history": list(history),
            "growth_cursor": cursor,
        }
        if self.artifact_store is None:
            return
        candidate_id = self.artifact_store.record_candidate(
            index=index,
            phase=phase,
            architecture=list(model.hidden_widths),
            score=score,
            accepted=accepted,
            context=self.context,
            candidate_best=stage.best_snapshot,
            candidate_last=stage.last_snapshot or stage.best_snapshot,
        )
        self.artifact_store.record_progress(
            candidate_id=candidate_id,
            context=self.context,
            candidate_score=score,
            candidate_accepted=accepted,
            global_best=selected.snapshot(cursor=selected_cursor),
        )

    def _validate_custom_candidate(self, current: TabularMLP, candidate: TabularMLP | None) -> TabularMLP | None:
        if candidate is None:
            return None
        if not isinstance(candidate, TabularMLP):
            raise TypeError("generic mutation functions must return TabularMLP or None")
        if candidate is current or candidate.hidden_widths == current.hidden_widths:
            raise ValueError("generic mutation function must change the architecture")
        if (candidate.input_dim != current.input_dim or candidate.output_dim != current.output_dim
                or len(candidate.hidden_widths) > self.settings.max_depth
                or any(width > self.settings.max_width for width in candidate.hidden_widths)):
            raise ValueError("generic mutation function returned an architecture outside model/capacity bounds")
        return candidate

    def _stage(self, model, arrays, *, resume_state=None):
        return self._stage_runner(model, arrays, generic=True, resume_state=resume_state, search_context=dict(self.context))

    def run(
        self,
        initial: TabularMLP,
        arrays,
        config: ModelConfig,
        *,
        resume_payload=None,
        artifact_resume_context: dict[str, Any] | None = None,
    ) -> SearchResult:
        settings = self.settings
        phase_orders = {
            "width": ("width",),
            "depth": ("depth",),
            "width_then_depth": ("width", "depth"),
            "depth_then_width": ("depth", "width"),
        }
        try:
            phases = phase_orders[config.order]
        except KeyError as exc:
            raise ValueError(
                f"generic candidate search does not support order {config.order!r}"
            ) from exc
        resume_context = dict(resume_payload.get("search_context", {})) if resume_payload else None
        resume_stage = resume_payload.get("stage_state") if resume_payload else None
        current = initial
        failure_counts: dict[str, int] = {}
        last_candidate_id = "candidate_000000"
        if resume_context is None:
            if artifact_resume_context is not None:
                resume_context = dict(artifact_resume_context)
                if resume_context.get("mutation_id", "builtin") != self.mutation_id:
                    raise ValueError("exact-resume failure: generic mutation identity does not match artifact state")
                current_snapshot = resume_context.pop("artifact_current_snapshot", None)
                if not isinstance(current_snapshot, dict):
                    raise ValueError("exact-resume failure: candidate artifact lacks current candidate weights")
                current = from_snapshot_preserving_rng(current_snapshot)
                if list(current.hidden_widths) != list(resume_context.get("current_architecture", [])):
                    raise ValueError("exact-resume failure: artifact current architecture does not match candidate state")
                snapshot = resume_context.get("global_best_snapshot")
                if not isinstance(snapshot, dict):
                    raise ValueError("exact-resume failure: candidate artifact lacks global-best snapshot")
                selected = from_snapshot_preserving_rng(snapshot)
                selected_score = float(resume_context["global_best_score"])
                current_score = float(resume_context["current_score"])
                previous_candidate_score = float(resume_context["previous_candidate_score"])
                history = list(resume_context.get("history", []))
                candidate_count = int(resume_context.get("candidate_count", 0))
                failure_counts = dict(resume_context.get("failure_counts", {}))
                resume_failures = int(resume_context.get("current_failures", 0))
                phase_name = str(resume_context.get("phase", "initial"))
                if phase_name == "initial":
                    phase_start = 0
                else:
                    phase_start = int(resume_context.get("phase_index", phases.index(phase_name)))
                self.context = resume_context
                last_candidate_id = self.artifact_store._candidate_name(candidate_count) if self.artifact_store else last_candidate_id
            else:
                self.context = {"controller": config.id, "phase": "initial", "profile": self.profile,
                                "mutation_id": self.mutation_id,
                                "current_architecture": list(current.hidden_widths), "history": []}
                stage = self._stage(current, arrays)
                current_score = stage.best_score
                selected = from_snapshot_preserving_rng(stage.best_snapshot)
                selected_score = current_score
                previous_candidate_score = current_score
                history = [{"architecture": list(current.hidden_widths), "score": current_score, "accepted": True, "stage": "initial"}]
                candidate_count = 0
                phase_start = 0
                resume_failures = 0
                self._commit_candidate(
                    index=0,
                    phase="initial",
                    model=current,
                    stage=stage,
                    score=current_score,
                    accepted=True,
                    selected=selected,
                    selected_score=selected_score,
                    current_score=current_score,
                    previous_candidate_score=previous_candidate_score,
                    failures=0,
                    failure_counts=failure_counts,
                    phase_index=0,
                    history=history,
                    candidate_count=0,
                )
                last_candidate_id = "candidate_000000"
        else:
            phase = str(resume_context.get("phase", ""))
            if resume_context.get("mutation_id", "builtin") != self.mutation_id:
                raise ValueError("exact-resume failure: generic mutation identity does not match checkpoint")
            if phase == "initial":
                self.context = resume_context
                stage = self._stage(current, arrays, resume_state=resume_stage)
                current_score = stage.best_score
                selected = from_snapshot_preserving_rng(stage.best_snapshot)
                selected_score = current_score
                previous_candidate_score = current_score
                history = [{"architecture": list(current.hidden_widths), "score": current_score, "accepted": True, "stage": "initial"}]
                candidate_count = 0
                phase_start = 0
                resume_failures = 0
                self._commit_candidate(
                    index=0,
                    phase="initial",
                    model=current,
                    stage=stage,
                    score=current_score,
                    accepted=True,
                    selected=selected,
                    selected_score=selected_score,
                    current_score=current_score,
                    previous_candidate_score=previous_candidate_score,
                    failures=0,
                    failure_counts=failure_counts,
                    phase_index=0,
                    history=history,
                    candidate_count=0,
                )
            elif phase in phases:
                phase_start = phases.index(phase)
                self.context = resume_context
                stage = self._stage(current, arrays, resume_state=resume_stage)
                snapshot = resume_context.get("global_best_snapshot")
                if not isinstance(snapshot, dict):
                    raise ValueError("exact-resume failure: generic stage lacks global-best snapshot")
                selected = from_snapshot_preserving_rng(snapshot)
                selected_score = float(resume_context["global_best_score"])
                current_score = float(resume_context["current_score"])
                previous_candidate_score = float(resume_context["previous_candidate_score"])
                history = list(resume_context.get("history", []))
                candidate_count = int(resume_context.get("candidate_count", 0))
                failure_counts = dict(resume_context.get("failure_counts", {}))
                resume_failures = int(resume_context.get("current_failures", 0))
                phase = phases[phase_start]
                candidate_score = stage.best_score
                margin = settings.delta_width if phase == "width" else settings.delta_depth
                if self.profile == "faithful":
                    stage_best = min(current_score, candidate_score)
                    improved_phase = candidate_score < stage_best - margin
                    accepted = candidate_score < previous_candidate_score
                    if accepted:
                        selected = from_snapshot_preserving_rng(stage.best_snapshot)
                    selected_score = candidate_score
                    current_score = candidate_score
                    previous_candidate_score = candidate_score
                    resume_failures = 0 if improved_phase else resume_failures + 1
                else:
                    improved_phase = candidate_score < current_score - margin
                    accepted = candidate_score < selected_score - margin
                    if accepted:
                        selected = from_snapshot_preserving_rng(stage.best_snapshot)
                        selected_score = candidate_score
                    if improved_phase:
                        current_score = candidate_score
                        resume_failures = 0
                    else:
                        resume_failures += 1
                    previous_candidate_score = candidate_score
                failure_counts[phase] = resume_failures
                history.append({"architecture": list(current.hidden_widths), "score": candidate_score, "accepted": accepted, "stage": phase, "profile": self.profile})
                self._commit_candidate(
                    index=candidate_count,
                    phase=phase,
                    model=current,
                    stage=stage,
                    score=candidate_score,
                    accepted=accepted,
                    selected=selected,
                    selected_score=selected_score,
                    current_score=current_score,
                    previous_candidate_score=previous_candidate_score,
                    failures=resume_failures,
                    failure_counts=failure_counts,
                    phase_index=phase_start,
                    history=history,
                    candidate_count=candidate_count,
                )
                last_candidate_id = self.artifact_store._candidate_name(candidate_count) if self.artifact_store else last_candidate_id
            else:
                raise ValueError(f"exact-resume failure: unsupported generic search phase {phase!r}")

        for phase_index in range(phase_start, len(phases)):
            phase = phases[phase_index]
            failures = resume_failures if phase_index == phase_start else 0
            resume_failures = 0
            patience = settings.patience_width_exp if phase == "width" else settings.patience_depth_exp
            while failures < patience:
                if settings.max_candidates and candidate_count >= settings.max_candidates:
                    break
                if phase == "width":
                    if self.widen_fn is not None:
                        candidate = self._validate_custom_candidate(
                            current, self.widen_fn(current, settings),
                        )
                    elif config.width_strategy == "all_layers":
                        candidate = expand_width_all_layers(current, settings.ex_k_width, settings.max_width)
                    else:
                        # Generic adp_search advances the first eligible layer under max_width.
                        candidate = expand_width(current, settings.ex_k_width, settings.max_width, strategy="round_robin")
                    if candidate is not None and self.profile == "corrected":
                        added_units = sum(candidate.hidden_widths) - sum(current.hidden_widths)
                        if total_neurons(current) + added_units > settings.max_neurons:
                            candidate = None
                else:
                    if self.deepen_fn is not None:
                        candidate = self._validate_custom_candidate(
                            current, self.deepen_fn(current, settings),
                        )
                    elif config.depth_seed == "current":
                        candidate = expand_depth_match_width(current, settings.max_depth)
                    else:
                        candidate = expand_depth(
                            current, settings.max_depth, settings.min_new_layer_width,
                        )
                    if candidate is not None and self.profile == "corrected":
                        if total_neurons(candidate) > settings.max_neurons:
                            candidate = None
                if candidate is None:
                    break
                candidate_count += 1
                self.context = {
                    "controller": config.id, "phase": phase, "profile": self.profile,
                    "mutation_id": self.mutation_id,
                    "current_architecture": list(candidate.hidden_widths),
                    "global_best_architecture": list(selected.hidden_widths),
                    "global_best_score": selected_score,
                    "global_best_snapshot": selected.snapshot(cursor=int(getattr(selected, "_width_cursor", 0))),
                    "candidate_count": candidate_count,
                    "counters": {f"{phase}_failures": failures},
                    "current_failures": failures,
                    "current_score": current_score,
                    "previous_candidate_score": previous_candidate_score,
                    "failure_counts": dict(failure_counts),
                    "phase_index": phase_index,
                    "history": list(history),
                    "growth_cursor": int(getattr(candidate, "_width_cursor", 0)),
                }
                stage = self._stage(candidate, arrays)
                current = candidate
                candidate_score = stage.best_score
                if self.profile == "faithful":
                    # Preserve source ordering: update the stage minimum before testing it.
                    stage_best = min(current_score, candidate_score)
                    improved_phase = candidate_score < stage_best - (settings.delta_width if phase == "width" else settings.delta_depth)
                    previous_candidate_improved = candidate_score < previous_candidate_score
                    if previous_candidate_improved:
                        selected = from_snapshot_preserving_rng(stage.best_snapshot)
                    # Source updates this value even when the candidate is not globally best.
                    selected_score = candidate_score
                    current_score = candidate_score
                    previous_candidate_score = candidate_score
                    failures = 0 if improved_phase else failures + 1
                    accepted = previous_candidate_improved
                else:
                    margin = settings.delta_width if phase == "width" else settings.delta_depth
                    improved_phase = candidate_score < current_score - margin
                    accepted = candidate_score < selected_score - margin
                    if accepted:
                        selected = from_snapshot_preserving_rng(stage.best_snapshot)
                        selected_score = candidate_score
                    if improved_phase:
                        current_score = candidate_score
                        failures = 0
                    else:
                        failures += 1
                    previous_candidate_score = candidate_score
                failure_counts[phase] = failures
                history.append({"architecture": list(current.hidden_widths), "score": candidate_score, "accepted": accepted, "stage": phase, "profile": self.profile})
                self._commit_candidate(
                    index=candidate_count,
                    phase=phase,
                    model=current,
                    stage=stage,
                    score=candidate_score,
                    accepted=accepted,
                    selected=selected,
                    selected_score=selected_score,
                    current_score=current_score,
                    previous_candidate_score=previous_candidate_score,
                    failures=failures,
                    failure_counts=failure_counts,
                    phase_index=phase_index,
                    history=history,
                    candidate_count=candidate_count,
                )
                last_candidate_id = self.artifact_store._candidate_name(candidate_count) if self.artifact_store else last_candidate_id
            if settings.max_candidates and candidate_count >= settings.max_candidates:
                break
        result_state = {
            "candidate_count": candidate_count,
            "profile": self.profile,
            "failure_counts": failure_counts,
            "current_architecture": list(current.hidden_widths),
            "best_architecture": list(selected.hidden_widths),
            "growth_cursor": int(getattr(current, "_width_cursor", 0)),
        }
        if self.artifact_store is not None:
            self.artifact_store.record_progress(
                candidate_id=last_candidate_id,
                context=self.context,
                candidate_score=float(history[-1]["score"]),
                candidate_accepted=bool(history[-1].get("accepted", False)),
                global_best=selected.snapshot(cursor=int(getattr(selected, "_width_cursor", 0))),
                task_complete=True,
                task_score=selected_score,
                task_history=history,
                task_state=result_state,
            )
        return SearchResult(selected, selected_score, history, result_state)
