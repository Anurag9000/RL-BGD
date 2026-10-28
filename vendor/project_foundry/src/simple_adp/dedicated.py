"""Dedicated patience-based DEN, nested DEN, and greedy LG-ND controllers."""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Sequence

import torch

from .checkpoint import stable_signature
from .config import ModelConfig, SearchSettings
from .growth import (
    can_add_depth, can_grow_width_all_layers, can_grow_width_round_robin,
    expand_depth, expand_depth_match_width, expand_width_all_layers,
    expand_width_den_round_robin, total_neurons,
)
from .models import TabularMLP, from_snapshot_preserving_rng, rebuild_mlp
from .results import SearchResult


class DedicatedController:
    """Implements the stage-based MSE/Adam DEN and LG-ND search policies."""

    def __init__(self, settings: SearchSettings, stage_runner: Callable) -> None:
        self.settings = settings
        self._stage_runner = stage_runner
        self.context: dict[str, object] = {}

    def _stage(self, model, arrays, *, resume_state=None):
        return self._stage_runner(
            model, arrays, generic=False, resume_state=resume_state,
            search_context=dict(self.context),
        )

    def run(self, initial: TabularMLP, arrays, config: ModelConfig, *, fixed_grid=None, resume_payload=None) -> SearchResult:
        if config.controller == "lgnd":
            return self._width_search(initial, arrays, config, greedy=True, resume_payload=resume_payload)
        if config.controller == "width_den":
            return self._width_search(initial, arrays, config, greedy=False, resume_payload=resume_payload)
        if config.controller == "depth_den":
            return self._depth_search(initial, arrays, config, resume_payload=resume_payload)
        if config.controller == "fixed":
            if fixed_grid is None or len(fixed_grid) == 0:
                raise ValueError("MLP-Fixed requires a non-empty fixed_grid of hidden-width vectors")
            return self._fixed_search(initial, arrays, fixed_grid, resume_payload=resume_payload)
        raise ValueError(f"unsupported dedicated controller {config.controller!r}")

    def _width_search(self, initial, arrays, config, *, greedy: bool, resume_payload=None) -> SearchResult:
        settings = self.settings
        resume_context = dict(resume_payload.get("search_context", {})) if resume_payload else None
        resume_stage = resume_payload.get("stage_state") if resume_payload else None
        stop_greedy_after_resumed_failure = False
        if resume_payload is not None and resume_stage is None and "hidden_widths" in resume_payload:
            # SearchController callbacks expose the stage record directly; the
            # estimator's durable envelope nests the same record under this key.
            resume_stage = resume_payload
        if resume_context is None:
            self.context = {"controller": config.id, "phase": "initial", "current_architecture": list(initial.hidden_widths), "history": []}
            stage = self._stage(initial, arrays)
            current = initial
            best = from_snapshot_preserving_rng(stage.best_snapshot)
            best_score = stage.best_score
            history = [{"architecture": list(current.hidden_widths), "score": best_score, "accepted": True, "stage": "initial"}]
            failures = 0
        else:
            phase = str(resume_context.get("phase", "initial"))
            self.context = resume_context
            stage = self._stage(initial, arrays, resume_state=resume_stage)
            current = initial
            history = list(resume_context.get("history", []))
            if phase == "initial":
                best = from_snapshot_preserving_rng(stage.best_snapshot)
                best_score = stage.best_score
                history.append({"architecture": list(current.hidden_widths), "score": best_score, "accepted": True, "stage": "initial"})
                failures = 0
            elif phase == "width":
                snapshot = resume_context.get("global_best_snapshot")
                if not isinstance(snapshot, dict):
                    raise ValueError("width-stage checkpoint lacks global-best snapshot")
                best = from_snapshot_preserving_rng(snapshot)
                best_score = float(resume_context["global_best_score"])
                failures = int(resume_context.get("counters", {}).get("width_failures", 0))
                accepted = stage.best_score < best_score if greedy else stage.best_score < best_score - settings.delta_width
                history.append({"architecture": list(current.hidden_widths), "score": stage.best_score, "accepted": accepted, "stage": "width"})
                if accepted:
                    best_score = stage.best_score
                    best = from_snapshot_preserving_rng(stage.best_snapshot)
                    failures = 0
                else:
                    failures += 1
                    stop_greedy_after_resumed_failure = greedy
            else:
                raise ValueError(f"cannot resume width controller from phase {phase!r}")
        while ((greedy and not stop_greedy_after_resumed_failure)
               or (not greedy and failures < settings.patience_width_exp)):
            if greedy:
                # The source LG-ND helper estimates capacity as if all layers
                # grew, even when a caller manually selects RR mutation.
                can_grow = can_grow_width_all_layers(
                    current, settings.ex_k_width, settings.max_width, settings.max_neurons,
                )
            elif config.width_strategy == "all_layers":
                can_grow = can_grow_width_all_layers(
                    current, settings.ex_k_width, settings.max_width, settings.max_neurons,
                )
            else:
                can_grow = can_grow_width_round_robin(
                    current, settings.ex_k_width, settings.max_width, settings.max_neurons,
                )
            if not can_grow:
                break
            if config.width_strategy == "all_layers":
                candidate = expand_width_all_layers(current, settings.ex_k_width, settings.max_width)
            else:
                candidate = expand_width_den_round_robin(current, settings.ex_k_width, settings.max_width)
            if candidate is None:
                break
            self.context = {
                "controller": config.id, "phase": "width",
                "current_architecture": list(candidate.hidden_widths),
                "global_best_architecture": list(best.hidden_widths),
                "global_best_score": best_score,
                "global_best_snapshot": best.snapshot(cursor=int(getattr(best, "_width_cursor", 0))),
                "counters": {"width_failures": failures},
                "growth_cursor": int(getattr(candidate, "_width_cursor", 0)),
                "history": list(history),
            }
            stage = self._stage(candidate, arrays)
            current = candidate
            improvement = stage.best_score < best_score if greedy else stage.best_score < best_score - settings.delta_width
            history.append({"architecture": list(current.hidden_widths), "score": stage.best_score, "accepted": improvement, "stage": "width"})
            if improvement:
                best_score = stage.best_score
                best = from_snapshot_preserving_rng(stage.best_snapshot)
                failures = 0
            else:
                failures += 1
                if greedy:
                    break
        return SearchResult(best, best_score, history, {
            "width_failures": failures,
            "current_architecture": list(current.hidden_widths),
            "best_architecture": list(best.hidden_widths),
            "growth_cursor": int(getattr(current, "_width_cursor", 0)),
        })

    def _depth_search(self, initial, arrays, config, *, resume_payload=None) -> SearchResult:
        settings = self.settings
        resume_context = dict(resume_payload.get("search_context", {})) if resume_payload else None
        resume_stage = resume_payload.get("stage_state") if resume_payload else None
        resumed_progressive = False
        progressive_start_width = 0
        if resume_context is None:
            self.context = {"controller": config.id, "phase": "initial", "current_architecture": list(initial.hidden_widths), "history": []}
            stage = self._stage(initial, arrays)
            current = initial
            best = from_snapshot_preserving_rng(stage.best_snapshot)
            best_score = stage.best_score
            history = [{"architecture": list(current.hidden_widths), "score": best_score, "accepted": True, "stage": "initial"}]
            failures = 0
        else:
            phase = str(resume_context.get("phase", "initial"))
            self.context = resume_context
            stage = self._stage(initial, arrays, resume_state=resume_stage)
            current = initial
            history = list(resume_context.get("history", []))
            if phase == "initial":
                best = from_snapshot_preserving_rng(stage.best_snapshot)
                best_score = stage.best_score
                history.append({"architecture": list(current.hidden_widths), "score": best_score, "accepted": True, "stage": "initial"})
                failures = 0
            elif phase in {"depth", "progressive_depth_step"}:
                snapshot = resume_context.get("global_best_snapshot")
                if not isinstance(snapshot, dict):
                    raise ValueError("depth-stage checkpoint lacks global-best snapshot")
                best = from_snapshot_preserving_rng(snapshot)
                best_score = float(resume_context["global_best_score"])
                failures = int(resume_context.get("counters", {}).get("depth_failures", 0))
                if phase == "progressive_depth_step":
                    resumed_progressive = True
                    progressive_start_width = int(resume_context["progressive_start_width"])
                else:
                    accepted = stage.best_score < best_score - settings.delta_depth
                    history.append({"architecture": list(current.hidden_widths), "score": stage.best_score, "accepted": accepted, "stage": "depth"})
                    if accepted:
                        best_score = stage.best_score
                        best = from_snapshot_preserving_rng(stage.best_snapshot)
                        failures = 0
                    else:
                        failures += 1
            else:
                raise ValueError(f"cannot resume depth controller from phase {phase!r}")
        while failures < settings.patience_depth_exp:
            if resumed_progressive:
                candidate = current
                while candidate.hidden_widths[-1] < progressive_start_width:
                    next_widths = list(candidate.hidden_widths)
                    next_widths[-1] += 1
                    if total_neurons(candidate) + 1 > settings.max_neurons:
                        current = candidate
                        candidate = None
                        break
                    candidate = rebuild_mlp(candidate, next_widths)
                    if candidate.hidden_widths[-1] < progressive_start_width:
                        self.context = {
                            "controller": config.id, "phase": "progressive_depth_step",
                            "current_architecture": list(candidate.hidden_widths),
                            "global_best_architecture": list(best.hidden_widths),
                            "global_best_score": best_score,
                            "global_best_snapshot": best.snapshot(cursor=int(getattr(best, "_width_cursor", 0))),
                            "counters": {"depth_failures": failures},
                            "progressive_start_width": progressive_start_width,
                            "depth_stage_bounds": {
                                "stage_index": len(candidate.hidden_widths) - 1,
                                "start_width": 1,
                                "target_width": progressive_start_width,
                            },
                            "history": list(history),
                        }
                        self._stage(candidate, arrays)
                resumed_progressive = False
                if candidate is None:
                    break
                self.context = {
                    "controller": config.id, "phase": "depth",
                    "current_architecture": list(candidate.hidden_widths),
                    "global_best_architecture": list(best.hidden_widths),
                    "global_best_score": best_score,
                    "global_best_snapshot": best.snapshot(cursor=int(getattr(best, "_width_cursor", 0))),
                    "counters": {"depth_failures": failures},
                    "history": list(history),
                }
                stage = self._stage(candidate, arrays)
                current = candidate
                accepted = stage.best_score < best_score - settings.delta_depth
                history.append({"architecture": list(current.hidden_widths), "score": stage.best_score, "accepted": accepted, "stage": "depth"})
                if accepted:
                    best_score = stage.best_score
                    best = from_snapshot_preserving_rng(stage.best_snapshot)
                    failures = 0
                else:
                    failures += 1
                continue
            seed_width = 1 if config.depth_seed == "one" else current.hidden_widths[-1]
            if not can_add_depth(current, settings.max_depth, settings.max_neurons, seed_width):
                break
            if config.progressive_depth:
                start_width = current.hidden_widths[-1]
                candidate = expand_depth(current, settings.max_depth, 1)
                if candidate is None:
                    break
                while candidate.hidden_widths[-1] < start_width:
                    self.context = {
                        "controller": config.id, "phase": "progressive_depth_step",
                        "current_architecture": list(candidate.hidden_widths),
                        "global_best_architecture": list(best.hidden_widths),
                        "global_best_score": best_score,
                        "global_best_snapshot": best.snapshot(cursor=int(getattr(best, "_width_cursor", 0))),
                        "counters": {"depth_failures": failures},
                        "progressive_start_width": start_width,
                        "depth_stage_bounds": {
                            "stage_index": len(candidate.hidden_widths) - 1,
                            "start_width": 1,
                            "target_width": start_width,
                        },
                        "history": list(history),
                    }
                    self._stage(candidate, arrays)  # optimization-only catch-up step
                    next_widths = list(candidate.hidden_widths)
                    next_widths[-1] += 1
                    if total_neurons(candidate) + 1 > settings.max_neurons:
                        current = candidate
                        candidate = None
                        break
                    candidate = rebuild_mlp(candidate, next_widths)
                if candidate is None:
                    break
            elif config.depth_seed == "current":
                candidate = expand_depth_match_width(current, settings.max_depth)
            else:
                candidate = expand_depth(current, settings.max_depth, 1)
            if candidate is None:
                break
            self.context = {
                "controller": config.id, "phase": "depth",
                "current_architecture": list(candidate.hidden_widths),
                "global_best_architecture": list(best.hidden_widths),
                "global_best_score": best_score,
                "global_best_snapshot": best.snapshot(cursor=int(getattr(best, "_width_cursor", 0))),
                "counters": {"depth_failures": failures},
                "history": list(history),
            }
            stage = self._stage(candidate, arrays)
            current = candidate
            accepted = stage.best_score < best_score - settings.delta_depth
            history.append({"architecture": list(current.hidden_widths), "score": stage.best_score, "accepted": accepted, "stage": "depth"})
            if accepted:
                best_score = stage.best_score
                best = from_snapshot_preserving_rng(stage.best_snapshot)
                failures = 0
            else:
                failures += 1
        return SearchResult(best, best_score, history, {
            "depth_failures": failures,
            "current_architecture": list(current.hidden_widths),
            "best_architecture": list(best.hidden_widths),
        })

    def _fixed_search(self, initial, arrays, fixed_grid: Sequence[Sequence[int]], *, resume_payload=None) -> SearchResult:
        resume_context = dict(resume_payload.get("search_context", {})) if resume_payload else None
        resume_stage = resume_payload.get("stage_state") if resume_payload else None
        best_model = from_snapshot_preserving_rng(resume_context["global_best_snapshot"]) if resume_context and resume_context.get("global_best_snapshot") else None
        best_score = float(resume_context.get("global_best_score", float("inf"))) if resume_context else float("inf")
        history = list(resume_context.get("history", [])) if resume_context else []
        start_index = int(resume_context.get("fixed_index", 0)) if resume_context else 0
        for index in range(start_index, len(fixed_grid)):
            widths = fixed_grid[index]
            architecture = [int(width) for width in widths]
            if not architecture or any(width <= 0 for width in architecture):
                raise ValueError("each fixed-grid architecture must be a non-empty vector of positive widths")
            model = TabularMLP(initial.input_dim, architecture, initial.output_dim).to(self.settings.device)
            self.context = {
                "controller": "MLP-Fixed", "phase": "fixed", "current_architecture": architecture,
                "global_best_snapshot": best_model.snapshot() if best_model is not None else None,
                "global_best_score": best_score, "fixed_index": index, "history": list(history),
            }
            stage = self._stage(model, arrays, resume_state=resume_stage if resume_payload and index == start_index else None)
            resume_stage = None
            history.append({"architecture": architecture, "score": stage.best_score, "accepted": stage.best_score < best_score, "stage": "fixed"})
            if stage.best_score < best_score:
                best_score = stage.best_score
                best_model = from_snapshot_preserving_rng(stage.best_snapshot)
        assert best_model is not None
        return SearchResult(best_model, best_score, history, {"fixed_grid": [list(widths) for widths in fixed_grid]})

    def run_nested_width_depth(
        self,
        initial,
        arrays,
        *,
        seed_new_depth_to_current_width: bool = False,
        width_growth_strategy: str = "all_layers",
        resume_payload=None,
        task_state=None,
    ) -> SearchResult:
        """Run and resume the guide's dedicated outer-width/inner-depth search."""
        settings = self.settings
        strategy_aliases = {
            "all_layers": "all_layers", "uniform": "all_layers", "simultaneous": "all_layers",
            "round_robin": "round_robin", "rr": "round_robin", "per_layer": "round_robin",
        }
        try:
            width_growth_strategy = strategy_aliases[width_growth_strategy]
        except KeyError as exc:
            raise ValueError(f"unknown width growth strategy {width_growth_strategy!r}") from exc
        resume_context = dict(resume_payload.get("search_context", {})) if resume_payload else None
        resume_stage = resume_payload.get("stage_state") if resume_payload else None
        if resume_payload is not None and resume_stage is None and "hidden_widths" in resume_payload:
            # SearchController callbacks expose the stage record directly; the
            # estimator's durable envelope nests the same record under this key.
            resume_stage = resume_payload
        resume_inner_round = False
        if resume_context is not None:
            initial_widths = [int(width) for width in resume_context.get(
                "search_initial_architecture", initial.hidden_widths,
            )]
        else:
            initial_widths = list(initial.hidden_widths)
        data_hash = hashlib.sha256()
        for value in arrays:
            tensor = torch.as_tensor(value).detach().cpu().contiguous()
            data_hash.update(str(tensor.dtype).encode("utf-8"))
            data_hash.update(str(tuple(tensor.shape)).encode("utf-8"))
            data_hash.update(tensor.numpy().tobytes())
        expected_signature = stable_signature({
            "controller": "nested_width_depth",
            "settings": settings.as_dict(),
            "initial_architecture": initial_widths,
            "seed_new_depth_to_current_width": seed_new_depth_to_current_width,
            "width_growth_strategy": width_growth_strategy,
            "task_state": task_state,
            "array_sha256": data_hash.hexdigest(),
        })

        if resume_context is None:
            current = initial
            history = []
            self.context = {
                "controller": "nested_width_depth", "phase": "nested_initial",
                "current_architecture": list(initial.hidden_widths), "history": [],
                "search_initial_architecture": initial_widths,
                "seed_new_depth_to_current_width": seed_new_depth_to_current_width,
                "width_growth_strategy": width_growth_strategy,
                "growth_cursor": int(getattr(initial, "_width_cursor", 0)),
                "controller_signature": expected_signature,
                "counters": {"width_failures": 0, "depth_failures": 0},
            }
            base = self._stage(initial, arrays)
            best = from_snapshot_preserving_rng(base.best_snapshot)
            best_score = base.best_score
            history.append({"architecture": list(current.hidden_widths), "score": best_score,
                            "accepted": True, "stage": "nested-initial"})
            width_failures = 0
        else:
            if resume_context.get("controller") != "nested_width_depth":
                raise ValueError("exact-resume failure: checkpoint controller identity does not match nested search")
            if resume_context.get("controller_signature") != expected_signature:
                raise ValueError("exact-resume failure: nested search settings/data signature does not match checkpoint")
            phase = resume_context.get("phase")
            if phase not in {"nested_initial", "inner_depth"}:
                raise ValueError(f"exact-resume failure: unsupported nested search phase {phase!r}")
            if not isinstance(resume_stage, dict):
                raise ValueError("exact-resume failure: nested search checkpoint lacks stage state")
            widths = [int(width) for width in resume_context.get("current_architecture", ())]
            if not widths or widths != list(initial.hidden_widths) or widths != list(resume_stage.get("hidden_widths", ())):
                raise ValueError("exact-resume failure: nested controller and stage architectures do not match")
            if resume_context.get("width_growth_strategy", "all_layers") != width_growth_strategy:
                raise ValueError("exact-resume failure: nested width growth strategy does not match checkpoint")
            setattr(initial, "_width_cursor", int(resume_context.get("growth_cursor", 0)))
            self.context = resume_context
            stage = self._stage(initial, arrays, resume_state=resume_stage)
            history = list(resume_context.get("history", []))
            if phase == "nested_initial":
                current = initial
                best = from_snapshot_preserving_rng(stage.best_snapshot)
                best_score = stage.best_score
                history.append({"architecture": list(current.hidden_widths), "score": best_score,
                                "accepted": True, "stage": "nested-initial"})
                width_failures = 0
            else:
                snapshot = resume_context.get("global_best_snapshot")
                if not isinstance(snapshot, dict):
                    raise ValueError("exact-resume failure: nested stage lacks global-best snapshot")
                best = from_snapshot_preserving_rng(snapshot)
                best_score = float(resume_context["global_best_score"])
                counters = resume_context.get("counters", {})
                width_failures = int(counters.get("width_failures", 0))
                depth_failures = int(counters.get("depth_failures", 0))
                before_round_best = float(resume_context["before_round_best"])
                current = initial
                accepted = stage.best_score < best_score - settings.delta_depth
                history.append({"architecture": list(current.hidden_widths), "score": stage.best_score,
                                "accepted": accepted, "stage": "nested-depth"})
                if accepted:
                    best_score = stage.best_score
                    best = from_snapshot_preserving_rng(stage.best_snapshot)
                    depth_failures = 0
                else:
                    depth_failures += 1
                resume_inner_round = True

        while width_failures < settings.patience_width_exp:
            if resume_inner_round:
                # The interrupted candidate has already been scored above.
                before_round_best = float(resume_context["before_round_best"])
                resume_inner_round = False
            else:
                can_grow = (
                    can_grow_width_all_layers(current, settings.ex_k_width, settings.max_width, settings.max_neurons)
                    if width_growth_strategy == "all_layers"
                    else can_grow_width_round_robin(current, settings.ex_k_width, settings.max_width, settings.max_neurons)
                )
                if not can_grow:
                    break
                before_round_best = best_score
                widened = (
                    expand_width_all_layers(current, settings.ex_k_width, settings.max_width)
                    if width_growth_strategy == "all_layers"
                    else expand_width_den_round_robin(current, settings.ex_k_width, settings.max_width)
                )
                if widened is None:
                    break
                current = widened
                depth_failures = 0

            while depth_failures < settings.patience_depth_exp:
                depth_seed_width = (
                    current.hidden_widths[-1]
                    if seed_new_depth_to_current_width else 1
                )
                if not can_add_depth(
                    current, settings.max_depth, settings.max_neurons, depth_seed_width,
                ):
                    break
                deeper = (
                    expand_depth_match_width(current, settings.max_depth)
                    if seed_new_depth_to_current_width
                    else expand_depth(current, settings.max_depth, 1)
                )
                if deeper is None:
                    break
                self.context = {
                    "controller": "nested_width_depth", "phase": "inner_depth",
                    "current_architecture": list(deeper.hidden_widths),
                    "search_initial_architecture": initial_widths,
                    "seed_new_depth_to_current_width": seed_new_depth_to_current_width,
                    "width_growth_strategy": width_growth_strategy,
                    "growth_cursor": int(getattr(deeper, "_width_cursor", 0)),
                    "controller_signature": expected_signature,
                    "global_best_architecture": list(best.hidden_widths),
                    "global_best_score": best_score,
                    "global_best_snapshot": best.snapshot(cursor=int(getattr(best, "_width_cursor", 0))),
                    "before_round_best": before_round_best,
                    "counters": {"width_failures": width_failures, "depth_failures": depth_failures},
                    "history": list(history),
                }
                stage = self._stage(deeper, arrays)
                current = deeper
                accepted = stage.best_score < best_score - settings.delta_depth
                history.append({"architecture": list(current.hidden_widths), "score": stage.best_score,
                                "accepted": accepted, "stage": "nested-depth"})
                if accepted:
                    best_score = stage.best_score
                    best = from_snapshot_preserving_rng(stage.best_snapshot)
                    depth_failures = 0
                else:
                    depth_failures += 1

            width_failures = 0 if before_round_best - best_score >= settings.delta_width else width_failures + 1

        return SearchResult(best, best_score, history, {
            "width_failures": width_failures,
            "current_architecture": list(current.hidden_widths),
            "best_architecture": list(best.hidden_widths),
            "growth_cursor": int(getattr(current, "_width_cursor", 0)),
            "width_growth_strategy": width_growth_strategy,
        })
