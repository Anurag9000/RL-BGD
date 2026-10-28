"""Task adapters decouple tabular objectives and predictions from search policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TaskOutputSpec:
    """MLP output dimensions plus adapter-defined interpretation metadata."""

    output_dim: int
    semantics: str


class TaskAdapter(ABC):
    """Contract for mapping task targets, losses, validation scores, and predictions."""

    @abstractmethod
    def fit_targets(self, targets: Any) -> None:
        """Learn target metadata from training targets only."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of values produced per row by the MLP."""

    @property
    def output_spec(self) -> TaskOutputSpec:
        """Output declaration; subclasses may override its semantic label."""
        return TaskOutputSpec(self.output_dim, type(self).__qualname__)

    @abstractmethod
    def encode_targets(self, targets: Any, *, device: torch.device) -> torch.Tensor:
        """Convert raw targets into the loss' expected tensor representation."""

    @abstractmethod
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return a scalar training objective; lower validation score is better."""

    @abstractmethod
    def validation_score(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Return a finite scalar minimized by architecture selection."""

    @abstractmethod
    def predictions(self, outputs: torch.Tensor) -> np.ndarray:
        """Convert model outputs to user-facing predictions."""

    def predict_proba(self, outputs: torch.Tensor) -> np.ndarray:
        raise TypeError(f"{type(self).__name__} does not define predict_proba")

    def state_dict(self) -> dict[str, object]:
        """Serializable task metadata included in model checkpoints."""
        return {"type": type(self).__qualname__}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("type") != type(self).__qualname__:
            raise ValueError("checkpoint task adapter does not match the active adapter")

    def auxiliary_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return outputs.new_zeros(())

    def auxiliary_loss_components(
        self, outputs: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return equality and inequality penalties for generic ADP search.

        The default maps the legacy aggregate ``auxiliary_loss`` hook to the
        equality component. Tasks with both penalty types should override this
        method and return the two unweighted scalar terms separately.
        """
        return self.auxiliary_loss(outputs, targets), outputs.new_zeros(())

    def adp_search_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        lambda_loss: float,
        lambda_eq: float,
        lambda_ineq: float,
    ) -> torch.Tensor:
        """Apply the guide's penalty-gated generic-search objective."""
        prediction = self.loss(outputs, targets)
        if lambda_eq == 0 and lambda_ineq == 0:
            # The source path bypasses lambda_loss when both penalty weights
            # are zero, yielding the pure prediction objective.
            return prediction
        equality, inequality = self.auxiliary_loss_components(outputs, targets)
        return (
            lambda_loss * prediction
            + lambda_eq * equality
            + lambda_ineq * inequality
        )

    def evaluation_metrics(
        self,
        targets: Any,
        predictions: Any,
        probabilities: Any | None = None,
    ) -> dict[str, float | int | None]:
        """Optional test-set metrics for suite workers; custom tasks may override."""
        return {"n_samples": int(len(targets))}


class RegressionTask(TaskAdapter):
    """Mean-squared-error regression adapter, matching the dedicated guide baseline."""

    def __init__(self) -> None:
        self._output_dim: int | None = None

    def fit_targets(self, targets: Any) -> None:
        values = np.asarray(targets)
        if values.ndim == 1:
            self._output_dim = 1
        elif values.ndim == 2 and values.shape[1] > 0:
            self._output_dim = int(values.shape[1])
        else:
            raise ValueError("regression targets must have shape (n,) or (n, outputs)")

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("fit_targets must be called before output_dim")
        return self._output_dim

    def encode_targets(self, targets: Any, *, device: torch.device) -> torch.Tensor:
        values = np.asarray(targets, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != self.output_dim:
            raise ValueError("regression target shape differs from training target shape")
        if not np.isfinite(values).all():
            raise ValueError("regression targets must be finite")
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(outputs, targets)

    def validation_score(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return float(nn.functional.mse_loss(outputs, targets).detach().cpu())

    def predictions(self, outputs: torch.Tensor) -> np.ndarray:
        values = outputs.detach().cpu().numpy()
        return values[:, 0] if self.output_dim == 1 else values

    def evaluation_metrics(
        self,
        targets: Any,
        predictions: Any,
        probabilities: Any | None = None,
    ) -> dict[str, float | int | None]:
        truth = np.asarray(targets, dtype=np.float64)
        predicted = np.asarray(predictions, dtype=np.float64)
        if truth.shape != predicted.shape:
            raise ValueError("regression predictions and targets must have matching shapes")
        if truth.ndim == 1:
            truth = truth[:, None]
            predicted = predicted[:, None]
        if truth.ndim != 2 or len(truth) == 0:
            raise ValueError("regression evaluation expects non-empty (n,) or (n, outputs) arrays")
        residual = truth - predicted
        mse = np.mean(np.square(residual), axis=0)
        mae = np.mean(np.abs(residual), axis=0)
        if len(truth) < 2:
            r2 = None
        else:
            total = np.sum(np.square(truth - np.mean(truth, axis=0)), axis=0)
            residual_sum = np.sum(np.square(residual), axis=0)
            per_output = np.ones_like(total)
            nonconstant = total != 0
            per_output[nonconstant] = 1.0 - residual_sum[nonconstant] / total[nonconstant]
            per_output[~nonconstant & (residual_sum != 0)] = 0.0
            r2 = float(np.mean(per_output))
        return {
            "rmse": float(np.sqrt(np.mean(mse))),
            "mae": float(np.mean(mae)),
            "r2": r2,
        }

    def state_dict(self) -> dict[str, object]:
        return {"type": type(self).__qualname__, "output_dim": self._output_dim}

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        dim = state.get("output_dim")
        self._output_dim = int(dim) if dim is not None else None


class ClassificationTask(TaskAdapter):
    """Single-label classification with cross-entropy and original-label predictions."""

    def __init__(self) -> None:
        self.classes_: list[object] = []
        self._class_to_index: dict[tuple[str, str], int] = {}

    @staticmethod
    def _key(value: object) -> tuple[str, str]:
        scalar = value.item() if isinstance(value, np.generic) else value
        return (type(scalar).__qualname__, repr(scalar))

    def fit_targets(self, targets: Any) -> None:
        values = np.asarray(targets)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("classification targets must be a non-empty one-dimensional array")
        by_key = {self._key(value): value.item() if isinstance(value, np.generic) else value for value in values}
        keys = sorted(by_key)
        if len(keys) < 2:
            raise ValueError("classification requires at least two classes")
        self.classes_ = [by_key[key] for key in keys]
        self._class_to_index = {key: i for i, key in enumerate(keys)}

    @property
    def output_dim(self) -> int:
        if not self.classes_:
            raise RuntimeError("fit_targets must be called before output_dim")
        return len(self.classes_)

    def encode_targets(self, targets: Any, *, device: torch.device) -> torch.Tensor:
        values = np.asarray(targets)
        if values.ndim != 1:
            raise ValueError("classification targets must be one-dimensional")
        try:
            encoded = [self._class_to_index[self._key(value)] for value in values]
        except KeyError as exc:
            raise ValueError(f"target class {exc.args[0]!r} was absent from training data") from exc
        return torch.as_tensor(encoded, dtype=torch.long, device=device)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(outputs, targets)

    def validation_score(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return float(nn.functional.cross_entropy(outputs, targets).detach().cpu())

    def predict_proba(self, outputs: torch.Tensor) -> np.ndarray:
        return torch.softmax(outputs, dim=1).detach().cpu().numpy()

    def predictions(self, outputs: torch.Tensor) -> np.ndarray:
        indexes = outputs.argmax(dim=1).detach().cpu().numpy()
        return np.asarray(self.classes_, dtype=object)[indexes]

    def evaluation_metrics(
        self,
        targets: Any,
        predictions: Any,
        probabilities: Any | None = None,
    ) -> dict[str, float | int | None]:
        truth = self.encode_targets(targets, device=torch.device("cpu")).numpy()
        predicted = self.encode_targets(predictions, device=torch.device("cpu")).numpy()
        if truth.shape != predicted.shape or truth.size == 0:
            raise ValueError("classification predictions and targets must have matching non-empty shapes")
        class_count = self.output_dim
        support = np.bincount(truth, minlength=class_count)
        predicted_count = np.bincount(predicted, minlength=class_count)
        true_positive = np.bincount(truth[truth == predicted], minlength=class_count)
        recall = np.divide(
            true_positive, support, out=np.zeros(class_count, dtype=np.float64), where=support > 0,
        )
        denominator = 2 * true_positive + (predicted_count - true_positive) + (support - true_positive)
        f1 = np.divide(
            2 * true_positive, denominator,
            out=np.zeros(class_count, dtype=np.float64), where=denominator > 0,
        )
        active_classes = (support + predicted_count) > 0
        metrics: dict[str, float | int | None] = {
            "accuracy": float(np.mean(truth == predicted)),
            "balanced_accuracy": float(np.mean(recall[support > 0])),
            "macro_f1": float(np.mean(f1[active_classes])),
        }
        if probabilities is not None:
            probs = np.asarray(probabilities, dtype=np.float64)
            if probs.shape != (len(truth), class_count):
                raise ValueError("classification probabilities do not match target and class dimensions")
            row_sums = probs.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0) or not np.isfinite(probs).all():
                raise ValueError("classification probabilities must be finite with positive row sums")
            probs = probs / row_sums
            epsilon = np.finfo(probs.dtype).eps
            probs = np.clip(probs, epsilon, 1.0 - epsilon)
            metrics["log_loss"] = float(-np.mean(np.log(probs[np.arange(len(truth)), truth])))
        return metrics

    def state_dict(self) -> dict[str, object]:
        return {"type": type(self).__qualname__, "classes": self.classes_}

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self.classes_ = list(state.get("classes", []))
        self._class_to_index = {self._key(value): i for i, value in enumerate(self.classes_)}


class CallableTask(TaskAdapter):
    """Adapter for custom objectives without requiring a new library subclass."""

    def __init__(
        self,
        output_dim: int,
        loss_fn,
        score_fn,
        prediction_fn,
        *,
        target_encoder=None,
        proba_fn=None,
        task_name: str = "callable",
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self._output_dim = int(output_dim)
        self.loss_fn = loss_fn
        self.score_fn = score_fn
        self.prediction_fn = prediction_fn
        self.target_encoder = target_encoder or self._default_encode
        self.proba_fn = proba_fn
        self.task_name = task_name

    def fit_targets(self, targets: Any) -> None:
        return None

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @staticmethod
    def _default_encode(targets: Any, device: torch.device) -> torch.Tensor:
        array = np.asarray(targets)
        dtype = torch.long if np.issubdtype(array.dtype, np.integer) else torch.float32
        return torch.as_tensor(array, dtype=dtype, device=device)

    def encode_targets(self, targets: Any, *, device: torch.device) -> torch.Tensor:
        return self.target_encoder(targets, device)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(outputs, targets)

    def validation_score(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return float(self.score_fn(outputs, targets))

    def predictions(self, outputs: torch.Tensor) -> np.ndarray:
        result = self.prediction_fn(outputs)
        return result.detach().cpu().numpy() if torch.is_tensor(result) else np.asarray(result)

    def predict_proba(self, outputs: torch.Tensor) -> np.ndarray:
        if self.proba_fn is None:
            return super().predict_proba(outputs)
        result = self.proba_fn(outputs)
        return result.detach().cpu().numpy() if torch.is_tensor(result) else np.asarray(result)

    def state_dict(self) -> dict[str, object]:
        return {"type": type(self).__qualname__, "task_name": self.task_name, "output_dim": self.output_dim}

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        if state.get("task_name") != self.task_name or int(state.get("output_dim", -1)) != self.output_dim:
            raise ValueError("checkpoint callable-task metadata does not match the active task")
