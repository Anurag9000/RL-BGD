"""Stage-level training and exact same-stage recovery."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from .checkpoint import capture_rng_state, restore_rng_state
from .config import SearchSettings
from .models import TabularMLP
from .tasks import TaskAdapter


@dataclass
class StageResult:
    best_score: float
    best_snapshot: dict[str, object]
    epochs: int
    batches: int
    checkpoint_state: dict[str, object]
    last_snapshot: dict[str, object] | None = None


class StageTrainer:
    """Trains one architecture; every fresh architecture stage gets fresh optimizer state."""

    def __init__(
        self,
        task: TaskAdapter,
        settings: SearchSettings,
        *,
        optimizer_kind: str = "adam",
        use_penalties: bool = False,
        checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if optimizer_kind not in {"adam", "adamw"}:
            raise ValueError("optimizer_kind must be adam or adamw")
        self.task = task
        self.settings = settings
        self.optimizer_kind = optimizer_kind
        self.use_penalties = use_penalties
        self.checkpoint_callback = checkpoint_callback

    def _optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self.optimizer_kind == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.settings.learning_rate,
                weight_decay=self.settings.weight_decay,
            )
        return torch.optim.Adam(model.parameters(), lr=self.settings.learning_rate)

    @staticmethod
    def _score(task: TaskAdapter, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            return task.validation_score(model(x), y)

    def train_stage(
        self,
        model: TabularMLP,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        *,
        resume_state: dict[str, object] | None = None,
    ) -> StageResult:
        settings = self.settings
        device = torch.device(settings.device)
        model.to(device)
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_valid, y_valid = x_valid.to(device), y_valid.to(device)
        if len(x_train) == 0 or len(x_valid) == 0:
            raise ValueError("training and validation splits must be non-empty")
        optimizer = self._optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=settings.scheduler_t0,
            T_mult=settings.scheduler_t_mult,
            eta_min=settings.scheduler_eta_min,
        ) if self.optimizer_kind == "adam" else None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(settings.seed)
        best_score = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        best_cursor = int(getattr(model, "_width_cursor", 0))
        patience = settings.patience_es
        epoch = 0
        batch_cursor = 0
        permutation: torch.Tensor | None = None
        batches_done = 0

        if resume_state is not None:
            try:
                if list(resume_state["hidden_widths"]) != model.hidden_widths:  # type: ignore[arg-type]
                    raise ValueError("stage checkpoint architecture does not match active model")
                model.load_state_dict(resume_state["model_state"])  # type: ignore[arg-type]
                optimizer.load_state_dict(resume_state["optimizer_state"])  # type: ignore[arg-type]
                if scheduler is not None:
                    scheduler_state = resume_state.get("scheduler_state")
                    if scheduler_state is None:
                        raise ValueError("stage checkpoint is missing scheduler state")
                    scheduler.load_state_dict(scheduler_state)  # type: ignore[arg-type]
                best_state = copy.deepcopy(resume_state["best_model_state"])
                best_score = float(resume_state["best_score"])
                best_cursor = int(resume_state.get("best_cursor", 0))
                patience = int(resume_state["patience"])
                epoch = int(resume_state["epoch"])
                batch_cursor = int(resume_state["batch_cursor"])
                permutation = resume_state.get("permutation")  # type: ignore[assignment]
                batches_done = int(resume_state.get("batches_done", 0))
                generator.set_state(resume_state["generator_state"])  # type: ignore[arg-type]
                rng_state = resume_state.get("rng_state")
                if not isinstance(rng_state, dict) or not all(
                    key in rng_state for key in ("python", "numpy", "torch")
                ):
                    raise ValueError("stage checkpoint is missing required RNG state")
                restore_rng_state(rng_state)  # type: ignore[arg-type]
            except Exception as exc:
                if "exact-resume failure:" in str(exc):
                    raise
                raise ValueError(f"exact-resume failure: stage checkpoint restore failed: {exc}") from exc

        total_batches = (len(x_train) + settings.batch_size - 1) // settings.batch_size

        def state_payload(active_epoch: int, active_batch: int, active_order: torch.Tensor | None) -> dict[str, object]:
            return {
                "hidden_widths": list(model.hidden_widths),
                "current_cursor": int(getattr(model, "_width_cursor", 0)),
                "model_state": copy.deepcopy(model.state_dict()),
                "last_model_state": copy.deepcopy(model.state_dict()),
                "best_model_state": copy.deepcopy(best_state),
                "best_score": best_score,
                "best_cursor": best_cursor,
                "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                "scheduler_state": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                "patience": patience,
                "epoch": active_epoch,
                "batch_cursor": active_batch,
                "permutation": active_order.clone() if active_order is not None else None,
                "generator_state": generator.get_state(),
                "rng_state": capture_rng_state(),
                "batches_done": batches_done,
                "stage_status": "active",
            }

        # Persist the new architecture/stage identity before its first batch.
        # This keeps the runner sidecar aligned with the checkpoint when a
        # pressure pause interrupts the very first epoch of a new candidate.
        if resume_state is None and self.checkpoint_callback is not None:
            self.checkpoint_callback(state_payload(epoch, batch_cursor, permutation))

        while patience > 0 and (settings.max_epochs == 0 or epoch < settings.max_epochs):
            if permutation is None:
                permutation = torch.randperm(len(x_train), generator=generator)
                batch_cursor = 0
            model.train()
            for batch_index in range(batch_cursor, total_batches):
                start = batch_index * settings.batch_size
                indexes = permutation[start : start + settings.batch_size].to(device)
                xb, yb = x_train.index_select(0, indexes), y_train.index_select(0, indexes)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(xb)
                if self.use_penalties:
                    loss = self.task.adp_search_loss(
                        prediction,
                        yb,
                        lambda_loss=settings.lambda_loss,
                        lambda_eq=settings.lambda_eq,
                        lambda_ineq=settings.lambda_ineq,
                    )
                else:
                    loss = self.task.loss(prediction, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                loss.backward()
                # The guide's optional gradient clipping belongs to the
                # generic AdamW/adp_search adapter. Dedicated DEN/LG-ND stages
                # use their local Adam loop without gradient clipping.
                if self.optimizer_kind == "adamw" and settings.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
                optimizer.step()
                batches_done += 1
                batch_cursor = batch_index + 1
                if self.checkpoint_callback is not None:
                    self.checkpoint_callback(state_payload(epoch, batch_cursor, permutation))

            if scheduler is not None:
                scheduler.step()
            score = self._score(self.task, model, x_valid, y_valid)
            if not torch.isfinite(torch.tensor(score)):
                raise FloatingPointError("non-finite validation score")
            if score < best_score - 1e-12:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_cursor = int(getattr(model, "_width_cursor", 0))
                patience = settings.patience_es
            else:
                patience -= 1
            epoch += 1
            permutation = None
            batch_cursor = 0
            if self.checkpoint_callback is not None:
                self.checkpoint_callback(state_payload(epoch, 0, None))

        last_state = copy.deepcopy(model.state_dict())
        last_cursor = int(getattr(model, "_width_cursor", 0))
        last_snapshot = {
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "hidden_widths": list(model.hidden_widths),
            "batch_norm": model.batch_norm,
            "cursor": last_cursor,
            "state_dict": last_state,
        }
        model.load_state_dict(best_state)
        setattr(model, "_width_cursor", best_cursor)
        final = state_payload(epoch, 0, None)
        final["stage_status"] = "completed"
        final["model_state"] = copy.deepcopy(model.state_dict())
        final["last_model_state"] = last_state
        final["best_model_state"] = copy.deepcopy(model.state_dict())
        final["best_score"] = best_score
        final["patience"] = patience
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(final)
        return StageResult(
            best_score,
            model.snapshot(cursor=best_cursor),
            epoch,
            batches_done,
            final,
            last_snapshot,
        )
