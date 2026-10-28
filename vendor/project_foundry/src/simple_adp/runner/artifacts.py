"""Canonical paths for artifacts shared by the runner and its workers."""

from __future__ import annotations

from pathlib import Path

ESTIMATOR_CHECKPOINT_FILENAME = "estimator.checkpoint.pt"
ESTIMATOR_MODEL_FILENAME = "estimator.model.pt"


def estimator_checkpoint_path(result_dir: str | Path) -> Path:
    """Return the signed checkpoint containing current and best snapshots."""
    return Path(result_dir).resolve() / ESTIMATOR_CHECKPOINT_FILENAME


def estimator_model_path(checkpoint_path: str | Path) -> Path:
    """Return the final model artifact path beside the checkpoint."""
    return Path(checkpoint_path).with_name(ESTIMATOR_MODEL_FILENAME)
