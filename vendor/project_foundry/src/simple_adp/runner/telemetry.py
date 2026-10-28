"""Live and offline progress counting with ledger-compatible precedence."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .artifacts import estimator_checkpoint_path

from .state import JobStateStore

_BATCH_DONE_RE = re.compile(r"\bbatch=\d+/\d+\s+done\b")
_EPOCH_DONE_RE = re.compile(r"\b(?:ES\s+)?epoch=\d+\b.*\b(?:val|validation|test|eval)=")


class ChildTelemetryWriter:
    """Write the generic child progress sidecars consumed by runner status views.

    This intentionally records orchestration progress only. Task metrics and
    predictions remain in the worker result artifact.
    """

    CSV_FIELDS = (
        "timestamp", "kind", "job_id", "model_id", "stage", "split", "epoch", "batch",
        "patience", "best_loss", "completed", "artifact",
    )

    def __init__(
        self,
        result_dir: str | Path,
        *,
        job_id: str,
        model_id: str,
        checkpoint: str | Path | None = None,
        checkpoint_mirrors: dict[str, str | Path] | None = None,
    ) -> None:
        self.root = Path(result_dir) / "telemetry" / "simple_adp" / job_id
        self.csv_path = self.root / "telemetry.csv"
        self.progress_path = self.root / "progress_state.json"
        self.stage_path = self.root / "stage_state.json"
        self.last_checkpoint_path = self.root / "checkpoint_last.pt"
        self.best_checkpoint_path = self.root / "checkpoint_best.pt"
        self.best_batch_checkpoint_path = self.root / "checkpoint_best_batch.pt"
        self.results_dir = Path(result_dir)
        self.job_id = job_id
        self.model_id = model_id
        self.checkpoint = str(checkpoint) if checkpoint is not None else ""
        self.checkpoint_mirrors = {
            name: Path(path)
            for name, path in (checkpoint_mirrors or {}).items()
            if name in {"last", "best", "best_batch"}
        }
        self._checkpoint_stage_key: str | None = None
        self._stage_rng_deferred_to_checkpoint = False
        self.batches_done, self.epochs_done = self._load_counters()
        self.root.mkdir(parents=True, exist_ok=True)

    def _load_counters(self) -> tuple[int, int]:
        try:
            progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if isinstance(progress, dict):
                # Match the child telemetry loader: persisted integer values are
                # authoritative as written, including negative values, and any
                # read/parse/conversion failure falls through to CSV recovery.
                return int(progress.get("history_rows") or 0), int(progress.get("final_epoch") or 0)
        except Exception:
            pass
        counts = count_csv_records(
            self.csv_path,
            {"batch", "epoch"},
            kind_precedence=("record_type", "kind"),
        )
        return counts["batch"], counts["epoch"]

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _atomic_torch_save(path: Path, payload: Any) -> None:
        """Replace a telemetry Torch file through its fixed sibling temp path.

        Runner telemetry uses atomic replacement but does not fsync. The
        estimator's authoritative exact-resume checkpoint has its own stronger
        durability protocol in ``CheckpointStore``.
        """
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def save_last_checkpoint(self, payload: Any, *, mirror_path: str | Path | None = None) -> None:
        """Write the telemetry-local last checkpoint and optional stage mirror."""
        self._atomic_torch_save(self.last_checkpoint_path, payload)
        if mirror_path is not None:
            self._atomic_torch_save(Path(mirror_path), payload)

    def save_best_checkpoint(self, payload: Any, *, mirror_path: str | Path | None = None) -> None:
        """Write the telemetry-local best checkpoint and optional stage mirror."""
        self._atomic_torch_save(self.best_checkpoint_path, payload)
        if mirror_path is not None:
            self._atomic_torch_save(Path(mirror_path), payload)

    def save_best_batch_checkpoint(self, payload: Any, *, mirror_path: str | Path | None = None) -> None:
        """Write the telemetry-local best-batch checkpoint and optional mirror."""
        self._atomic_torch_save(self.best_batch_checkpoint_path, payload)
        if mirror_path is not None:
            self._atomic_torch_save(Path(mirror_path), payload)

    def persist_training_checkpoint(self, payload: dict[str, Any]) -> None:
        """Mirror the authoritative per-batch and best-validation records.

        The signed estimator checkpoint remains authoritative for exact resume.
        These compatibility artifacts provide the runner's stage-level last
        and best paths. The canonical guide does not define a best-batch
        selection/update rule, so this method deliberately leaves that artifact
        to an adapter that has such a rule.
        """
        self.save_last_checkpoint(
            payload,
            mirror_path=self.checkpoint_mirrors.get("last"),
        )
        stage = payload.get("stage_state")
        if not isinstance(stage, dict):
            return
        context = payload.get("search_context")
        context = context if isinstance(context, dict) else {}
        stage_key = self._stage_key(context)
        new_stage = stage_key != self._checkpoint_stage_key
        self._checkpoint_stage_key = stage_key
        batch_cursor = int(stage.get("batch_cursor", 0) or 0)
        epoch = int(stage.get("epoch", 0) or 0)
        best_state = stage.get("best_model_state")
        if not isinstance(best_state, dict):
            return
        best_payload = copy.deepcopy(payload)
        best_stage = best_payload["stage_state"]
        best_stage["model_state"] = copy.deepcopy(best_state)
        best_stage["last_model_state"] = copy.deepcopy(best_state)
        if batch_cursor == 0:
            best_payload["stage_checkpoint_kind"] = "best_validation"
            self.save_best_checkpoint(
                best_payload,
                mirror_path=self.checkpoint_mirrors.get("best"),
            )
        if new_stage and epoch == 0 and batch_cursor == 0:
            # The documented initial checkpoint starts best-batch state from
            # the stage's initial architecture snapshot and infinite loss.
            # No subsequent replacement is made without a defined selection
            # rule for the best training batch.
            initial_payload = copy.deepcopy(best_payload)
            initial_payload["stage_checkpoint_kind"] = "initial_best_batch"
            initial_payload["best_batch_loss"] = float("inf")
            self.save_best_batch_checkpoint(
                initial_payload,
                mirror_path=self.checkpoint_mirrors.get("best_batch"),
            )

    def _save_stage_state(self, state: dict[str, Any], kind: str) -> None:
        """Persist a compact generic stage sidecar and separate RNG companion.

        The full, authoritative optimizer/scheduler/data cursor remains in the
        estimator checkpoint. This sidecar mirrors the controller's observable
        stage position and gives external status tools a stable JSON surface.
        """
        context = state.get("search_context")
        context = context if isinstance(context, dict) else {}
        stage_key = self._stage_key(context)
        rng_name = f"stage_state.rng.{time.time_ns()}.{uuid.uuid4().hex}.pt"
        rng_path = self.stage_path.parent / rng_name
        from ..checkpoint import capture_rng_state

        self._atomic_torch_save(rng_path, state.get("rng_state") or capture_rng_state())
        # Keep caller metadata on the sidecar, as the historical telemetry
        # store did, while keeping the binary RNG payload in its companion.
        stage = {key: value for key, value in state.items() if key != "rng_state"}
        stage.update({
            "schema": 2,
            "kind": "stage_state",
            "job_id": self.job_id,
            "model_id": self.model_id,
            "controller": context.get("controller"),
            "phase": context.get("phase"),
            "search_stage_key": stage_key,
            "depth_stage_bounds": context.get("depth_stage_bounds"),
            "active_stage": state.get("stage_status") == "active",
            "completed": state.get("stage_status") == "completed",
            "current_hidden_widths": state.get("hidden_widths", context.get("current_architecture")),
            "current_width_growth_cursor": state.get("current_cursor", context.get("growth_cursor", 0)),
            "epoch": state.get("epoch", 0),
            "batch_cursor": state.get("batch_cursor", 0),
            "batches_done": state.get("batches_done", 0),
            "patience": state.get("patience"),
            "best_loss": state.get("best_score"),
            "stage_checkpoint": self.checkpoint,
            "rng_checkpoint": rng_name,
            "rng_checkpoint_path": str(rng_path.relative_to(self.results_dir)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._atomic_json(self.stage_path, stage)
        row = dict(state)
        row["rng_checkpoint"] = rng_name
        self._append("stage_state", row)
        self._write_progress_state(state, kind)

    def _write_progress_state(self, state: dict[str, Any], kind: str) -> None:
        context = state.get("search_context")
        context = context if isinstance(context, dict) else {}
        self._atomic_json(self.progress_path, {
            "schema": 1,
            "job_id": self.job_id,
            "model_id": self.model_id,
            "history_rows": self.batches_done,
            "final_epoch": self.epochs_done,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "stage": context.get("phase", context.get("controller")),
            "split": "train",
            "epoch": int(state.get("epoch", 0) or 0),
            "batch": state.get("batch_cursor"),
            "batch_cursor": int(state.get("batch_cursor", 0) or 0),
            "patience": state.get("patience"),
            "architecture": state.get("hidden_widths", context.get("current_architecture")),
            "best_loss": state.get("best_score"),
            "checkpoint": self.checkpoint,
            "active_stage": state.get("stage_status") == "active",
            "artifact": None,
        })

    def load_stage_state(self) -> dict[str, Any] | None:
        """Load stage state using the ledger's sidecar RNG precedence rules."""
        self._stage_rng_deferred_to_checkpoint = False
        if not self.stage_path.exists():
            state = self._load_stage_state_from_csv()
            if state is None:
                return None
            return self._prepare_loaded_stage_state(state)
        try:
            state = json.loads(self.stage_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"exact-resume failure: stage sidecar is unreadable: {exc}") from exc
        if not isinstance(state, dict):
            raise ValueError("exact-resume failure: stage sidecar is not an object")
        return self._prepare_loaded_stage_state(state)

    def _prepare_loaded_stage_state(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Defer to an exact checkpoint or restore the legacy sidecar RNG."""
        if self._active_checkpoint_has_exact_state(state):
            self._stage_rng_deferred_to_checkpoint = True
            return state
        if state.get("active_stage") and not state.get("completed"):
            # An incomplete legacy active checkpoint belongs to controller
            # recovery. Do not treat the sidecar RNG as an exact stage state.
            checkpoint = state.get("stage_checkpoint")
            if checkpoint:
                candidate = Path(str(checkpoint)).resolve()
                expected = Path(self.checkpoint).resolve() if self.checkpoint else None
                if expected is not None and candidate == expected and not candidate.exists():
                    self._stage_rng_deferred_to_checkpoint = True
                    return state
            return None
        if state.get("current_hidden_widths") and state.get("stage_checkpoint"):
            # Rebuilding a saved architecture consumes RNG. The caller restores
            # the signed estimator checkpoint after that rebuild.
            self._stage_rng_deferred_to_checkpoint = True
            return state
        try:
            self._restore_stage_rng(state)
        except Exception as exc:
            raise ValueError(f"exact-resume failure: stage RNG state cannot be restored: {exc}") from exc
        return state

    def _active_checkpoint_has_exact_state(self, state: dict[str, Any]) -> bool:
        if not state.get("active_stage") or state.get("completed"):
            return False
        recorded = state.get("stage_checkpoint")
        expected = Path(self.checkpoint).resolve() if self.checkpoint else None
        if not recorded or expected is None:
            return False
        candidate = Path(str(recorded)).resolve()
        if candidate != expected or not candidate.is_file():
            return False
        try:
            import torch

            envelope = torch.load(candidate, map_location="cpu", weights_only=False)
            payload = envelope.get("payload") if isinstance(envelope, dict) else None
            stage = payload.get("stage_state") if isinstance(payload, dict) else None
            if not isinstance(stage, dict):
                return False
            required = (
                "model_state", "optimizer_state", "scheduler_state", "epoch",
                "batch_cursor", "rng_state", "best_model_state", "best_score",
                "patience",
            )
            rng = stage.get("rng_state")
            return (
                all(key in stage for key in required)
                and isinstance(rng, dict)
                and all(key in rng for key in ("python", "numpy", "torch"))
            )
        except Exception:
            return False

    def _restore_stage_rng(self, state: dict[str, Any]) -> None:
        """Restore a safe RNG companion path or a unique legacy local filename."""
        raw_path = state.get("rng_checkpoint_path")
        if raw_path:
            relative = Path(str(raw_path))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("stage RNG path must remain inside the results directory")
            candidate = (self.results_dir / relative).resolve()
            try:
                candidate.relative_to(self.results_dir.resolve())
            except ValueError as exc:
                raise ValueError("stage RNG path escapes the results directory") from exc
        else:
            name = state.get("rng_checkpoint")
            if not name or Path(str(name)).name != str(name):
                raise ValueError("stage state lacks a safe RNG companion filename")
            matches = sorted(self.results_dir.rglob(str(name))) if self.results_dir.exists() else []
            if len(matches) != 1:
                raise ValueError("stage RNG companion is missing or ambiguous")
            candidate = matches[0].resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"stage RNG companion is missing: {candidate}")
        import torch
        from ..checkpoint import restore_rng_state

        rng_state = torch.load(candidate, map_location="cpu", weights_only=False)
        if not isinstance(rng_state, dict) or not all(
            key in rng_state for key in ("python", "numpy", "torch")
        ):
            raise ValueError("stage RNG companion is incomplete")
        restore_rng_state(rng_state)

    def _load_stage_state_from_csv(self) -> dict[str, Any] | None:
        """Recover the latest matching stage row when the JSON sidecar is absent.

        CSV is an observation fallback, not an exact checkpoint. Fields remain
        strings except for the source-compatible ``completed`` parser; model,
        optimizer, scheduler, sampler, and RNG restoration stays with the
        signature-validated estimator checkpoint.
        """
        try:
            with self.csv_path.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, UnicodeError, csv.Error):
            return None
        for row in reversed(rows):
            if row.get("kind") != "stage_state":
                continue
            if row.get("job_id") not in (None, "", self.job_id):
                continue
            if row.get("model_id") not in (None, "", self.model_id):
                continue
            recovered: dict[str, Any] = dict(row)
            recovered["completed"] = str(row.get("completed") or "").strip().lower() in {
                "1", "true", "yes", "y", "on",
            }
            recovered["kind"] = "stage_state"
            recovered["schema"] = "2"
            return recovered
        return None

    def exists(self) -> bool:
        """File-like compatibility check for consumers of stage-state records."""
        return self.load_stage_state() is not None

    def read_text(self) -> str:
        """Return the current stage state using the source's JSON text shape."""
        state = self.load_stage_state()
        if state is None:
            return ""
        return json.dumps(state, sort_keys=True, indent=2)

    def write_text(self, value: str) -> int:
        """Persist a stage record through the same sidecar/RNG write path.

        The historical file-like interface wrapped invalid JSON as a raw-text
        stage record. Valid JSON scalars were returned to the caller but did
        not replace the dictionary sidecar.
        """
        try:
            payload = json.loads(value)
        except Exception:
            payload = {"kind": "stage_state", "raw_text": value}
        if isinstance(payload, dict):
            self._save_stage_state(payload, "stage_state")
        return len(value)

    @staticmethod
    def _architecture(value: Any, *, label: str) -> list[int]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"exact-resume failure: {label} lacks a valid hidden-width vector")
        try:
            widths = [int(width) for width in value]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"exact-resume failure: {label} has an invalid hidden-width vector") from exc
        if any(width <= 0 for width in widths):
            raise ValueError(f"exact-resume failure: {label} has a non-positive hidden width")
        return widths

    @staticmethod
    def _state_shapes(value: Any) -> dict[str, tuple[int, ...]] | None:
        import torch

        if not isinstance(value, dict) or not value:
            return None
        if any(not isinstance(tensor, torch.Tensor) for tensor in value.values()):
            return None
        return {key: tuple(tensor.shape) for key, tensor in value.items()}

    def reconcile_checkpoint(self, payload: dict[str, Any]) -> None:
        """Fail closed on a conflicting active sidecar; checkpoint state owns RNG.

        The estimator checkpoint is already signature-validated by
        ``CheckpointStore`` before this method is called. A monotonic width
        catch-up is accepted only when the sidecar and checkpoint identify the
        same controller stage and the current/best tensor layouts agree.
        """
        if not self.stage_path.exists():
            return
        try:
            sidecar = json.loads(self.stage_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"exact-resume failure: stage sidecar is unreadable: {exc}") from exc
        if not isinstance(sidecar, dict):
            raise ValueError("exact-resume failure: stage sidecar is not an object")
        if sidecar.get("schema") != 2:
            raise ValueError("exact-resume failure: unsupported stage sidecar schema")
        if sidecar.get("job_id") != self.job_id or sidecar.get("model_id") != self.model_id:
            raise ValueError("exact-resume failure: stage sidecar identity does not match this job")

        expected_checkpoint = Path(self.checkpoint).resolve() if self.checkpoint else None
        recorded_checkpoint = sidecar.get("stage_checkpoint")
        if expected_checkpoint is None or not recorded_checkpoint:
            raise ValueError("exact-resume failure: stage sidecar lacks its estimator checkpoint path")
        recorded_path = Path(str(recorded_checkpoint)).resolve()
        try:
            recorded_path.relative_to(self.results_dir.resolve())
        except ValueError as exc:
            raise ValueError("exact-resume failure: stage checkpoint path escapes the results directory") from exc
        if recorded_path != expected_checkpoint:
            raise ValueError("exact-resume failure: stage sidecar points to a different checkpoint")

        if payload.get("search_status") == "completed":
            if not sidecar.get("completed"):
                raise ValueError("exact-resume failure: completed estimator checkpoint conflicts with active sidecar")
            return

        stage = payload.get("stage_state")
        if not isinstance(stage, dict):
            raise ValueError("exact-resume failure: estimator checkpoint lacks stage state")
        if stage.get("stage_status") not in {"active", "completed"}:
            raise ValueError("exact-resume failure: estimator checkpoint has invalid stage status")
        rng = stage.get("rng_state")
        if not isinstance(rng, dict) or not all(key in rng for key in ("python", "numpy", "torch")):
            raise ValueError("exact-resume failure: active estimator checkpoint lacks complete RNG state")
        checkpoint_widths = self._architecture(stage.get("hidden_widths"), label="estimator checkpoint")
        sidecar_widths = self._architecture(sidecar.get("current_hidden_widths"), label="stage sidecar")
        checkpoint_context = payload.get("search_context")
        if not isinstance(checkpoint_context, dict):
            raise ValueError("exact-resume failure: estimator checkpoint lacks controller context")
        for field in ("controller", "phase"):
            if sidecar.get(field) != checkpoint_context.get(field):
                raise ValueError(f"exact-resume failure: stage sidecar {field} does not match checkpoint")

        try:
            sidecar_epoch = int(sidecar.get("epoch", 0))
            checkpoint_epoch = int(stage.get("epoch", 0))
            sidecar_cursor = int(sidecar.get("batch_cursor", 0))
            checkpoint_cursor = int(stage.get("batch_cursor", 0))
            sidecar_width_cursor = int(sidecar.get("current_width_growth_cursor", 0))
            checkpoint_width_cursor = int(stage.get("current_cursor", 0))
            sidecar_batches = int(sidecar.get("batches_done", 0))
            checkpoint_batches = int(stage.get("batches_done", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("exact-resume failure: stage sidecar contains invalid cursors") from exc
        if sidecar_epoch > checkpoint_epoch or (sidecar_epoch == checkpoint_epoch and sidecar_cursor > checkpoint_cursor):
            raise ValueError("exact-resume failure: stage sidecar is ahead of estimator checkpoint")
        if sidecar_batches > checkpoint_batches:
            raise ValueError("exact-resume failure: stage sidecar batch count is ahead of checkpoint")

        reconciled = False
        if checkpoint_widths != sidecar_widths:
            best_shapes = self._state_shapes(stage.get("best_model_state"))
            current_shapes = self._state_shapes(stage.get("model_state"))
            same_stage = bool(sidecar.get("search_stage_key")) and (
                sidecar.get("search_stage_key") == self._stage_key(checkpoint_context)
            )
            monotonic_catchup = (
                len(checkpoint_widths) == len(sidecar_widths)
                and all(current >= previous for previous, current in zip(sidecar_widths, checkpoint_widths))
                and checkpoint_widths != sidecar_widths
            )
            depth_phase = checkpoint_context.get("phase") in {"depth", "progressive_depth_step"}
            depth_bounds = checkpoint_context.get("depth_stage_bounds")
            bounds_match = self._depth_recovery_is_bounded(
                sidecar.get("depth_stage_bounds"), depth_bounds,
                sidecar_widths, checkpoint_widths,
            ) if depth_phase else True
            if not (
                same_stage and monotonic_catchup and bounds_match
                and best_shapes is not None and best_shapes == current_shapes
            ):
                raise ValueError("exact-resume failure: stage sidecar architecture conflicts with checkpoint")
            sidecar["current_hidden_widths"] = checkpoint_widths
            sidecar["current_width_growth_cursor"] = checkpoint_width_cursor
            sidecar["reconciled_from_checkpoint"] = True
            reconciled = True
        elif sidecar_width_cursor != checkpoint_width_cursor:
            raise ValueError("exact-resume failure: stage sidecar width cursor conflicts with checkpoint")

        if reconciled:
            self._atomic_json(self.stage_path, sidecar)

    @staticmethod
    def _depth_recovery_is_bounded(
        sidecar_bounds: Any,
        checkpoint_bounds: Any,
        sidecar_widths: list[int],
        checkpoint_widths: list[int],
    ) -> bool:
        """Validate the narrow in-stage depth-growth reconciliation window.

        A checkpoint can be one or more monotonic width-growth steps ahead of
        the JSON sidecar after an interruption. For depth growth, accept that
        lag only when both records identify the same inserted layer and the
        saved and recovered widths stay within the recorded start/target band.
        """
        if not isinstance(sidecar_bounds, dict) or not isinstance(checkpoint_bounds, dict):
            return False
        required = ("stage_index", "start_width", "target_width")
        try:
            if any(key not in sidecar_bounds or key not in checkpoint_bounds for key in required):
                return False
            old = {key: int(sidecar_bounds[key]) for key in required}
            new = {key: int(checkpoint_bounds[key]) for key in required}
        except (TypeError, ValueError, OverflowError):
            return False
        if old != new:
            return False
        stage_index = new["stage_index"]
        start_width = new["start_width"]
        target_width = new["target_width"]
        if (
            stage_index < 0 or start_width <= 0 or target_width < start_width
            or len(sidecar_widths) != len(checkpoint_widths)
            or len(sidecar_widths) != stage_index + 1
        ):
            return False
        return all(
            all(start_width <= width <= target_width for width in widths)
            for widths in (sidecar_widths, checkpoint_widths)
        )

    @staticmethod
    def _stage_key(context: dict[str, Any]) -> str:
        identity = {
            key: context.get(key)
            for key in ("controller", "phase", "history", "counters", "fixed_index", "depth_index", "width_index")
            if key in context
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()

    def _append(self, kind: str, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        exists = self.csv_path.exists()
        fieldnames = self.CSV_FIELDS
        if exists:
            try:
                with self.csv_path.open("r", newline="", encoding="utf-8") as existing:
                    header = next(csv.reader(existing), None)
                if header:
                    # Keep appending to older telemetry schemas without
                    # shifting columns or rewriting historical rows.
                    fieldnames = tuple(header)
            except (OSError, UnicodeError, csv.Error):
                fieldnames = self.CSV_FIELDS
        context = state.get("search_context")
        context = context if isinstance(context, dict) else {}
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "job_id": self.job_id,
            "model_id": self.model_id,
            "stage": context.get("phase", context.get("controller", "")),
            "split": "train",
            "epoch": state.get("epoch", ""),
            # The copied source writes stage rows without an exact batch
            # cursor. Keep the generic batch observation for batch/epoch rows.
            "batch": state.get("batch_cursor", "") if kind != "stage_state" else "",
            "patience": state.get("patience", ""),
            "best_loss": state.get("best_score", ""),
            "completed": state.get("stage_status") == "completed",
            "artifact": "",
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})

    def __call__(self, state: dict[str, Any]) -> None:
        stage_status = str(state.get("stage_status", "active"))
        batch_cursor = int(state.get("batch_cursor", 0) or 0)
        epoch = int(state.get("epoch", 0) or 0)
        if batch_cursor > 0:
            kind = "batch"
            self.batches_done += 1
        elif stage_status == "active" and epoch > 0:
            kind = "epoch"
            self.epochs_done += 1
        else:
            kind = "state"
        self._append(kind, state)
        self._write_progress_state(state, kind)
        # Batch-level exact resume already lives in estimator.checkpoint.pt;
        # stage sidecars are refreshed at epoch/controller boundaries only.
        if kind in {"epoch", "state"}:
            self._save_stage_state(state, kind)


def telemetry_callback(
    path: str | Path | None,
    *,
    job_id: str,
    model_id: str,
    checkpoint_path: str | Path | None = None,
    checkpoint_mirrors: dict[str, str | Path] | None = None,
) -> Callable[[dict[str, object]], None] | None:
    """Create the worker callback when a result directory was configured."""
    if path is None:
        return None
    return ChildTelemetryWriter(
        path,
        job_id=job_id,
        model_id=model_id,
        checkpoint=checkpoint_path or estimator_checkpoint_path(path),
        checkpoint_mirrors=checkpoint_mirrors,
    )


def count_text_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return sum(1 for _ in stream)
    except Exception:
        return 0


def count_job_log_progress(path: Path) -> tuple[int, int]:
    batches = epochs = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                batches += bool(_BATCH_DONE_RE.search(line))
                epochs += bool(_EPOCH_DONE_RE.search(line))
    except Exception:
        return 0, 0
    return int(batches), int(epochs)


def count_csv_records(
    path: Path,
    wanted: set[str],
    *,
    kind_precedence: tuple[str, str] = ("kind", "record_type"),
) -> dict[str, int]:
    counts = {kind: 0 for kind in wanted}
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                kind = str(row.get(kind_precedence[0]) or row.get(kind_precedence[1]) or "").strip().lower()
                if kind in counts:
                    counts[kind] += 1
    except Exception:
        return {kind: 0 for kind in wanted}
    return counts


def summarize_offline(runs: Iterable[Any]) -> dict[str, int]:
    batches = epochs = telemetry_dirs = telemetry_jobs = 0
    for spec in runs:
        root = Path(spec.result_dir) / "telemetry"
        if not root.exists():
            continue
        telemetry_jobs += 1
        seen = False
        csv_files = [path for path in root.rglob("telemetry.csv") if path.is_file()]
        if csv_files:
            for path in csv_files:
                seen = True
                counts = count_csv_records(path, {"batch", "epoch"})
                batches += counts["batch"]
                epochs += counts["epoch"]
        else:
            for path in root.rglob("batch_metrics.jsonl"):
                if path.is_file():
                    seen = True
                    batches += count_text_lines(path)
            for path in root.rglob("epoch_metrics.jsonl"):
                if path.is_file():
                    seen = True
                    epochs += count_text_lines(path)
        telemetry_dirs += int(seen)
    return {
        "batches_done": batches,
        "epochs_done": epochs,
        "telemetry_jobs": telemetry_jobs,
        "telemetry_dirs": telemetry_dirs,
    }


def summarize_live(run_root: str | Path, runs: Iterable[Any], state_store: JobStateStore) -> dict[str, int]:
    totals = {key: 0 for key in ("batches_done", "epochs_done", "live_batches_done", "live_epochs_done", "telemetry_jobs", "telemetry_dirs")}
    for spec in runs:
        state = state_store.load(spec.identity.run_id)
        if not isinstance(state, dict):
            continue
        is_live = state.get("status") in {"running", "paused", "suspended"}
        telemetry_root = Path(spec.result_dir) / "telemetry"
        progress_files = [path for path in telemetry_root.rglob("progress_state.json") if path.is_file()] if telemetry_root.exists() else []
        if progress_files:
            totals["telemetry_jobs"] += 1
            totals["telemetry_dirs"] += 1
            latest = sorted(progress_files, key=lambda path: (path.stat().st_mtime_ns, str(path)))[-1]
            try:
                progress = json.loads(latest.read_text(encoding="utf-8"))
            except Exception:
                # Presence of a sidecar blocks log/state fallback even if malformed.
                continue
            if not isinstance(progress, dict):
                continue
            try:
                batches = int(progress.get("history_rows") or 0)
                totals["batches_done"] += batches
                totals["live_batches_done"] += batches if is_live else 0
            except Exception:
                pass
            try:
                epochs = int(progress.get("final_epoch") or 0)
                totals["epochs_done"] += epochs
                totals["live_epochs_done"] += epochs if is_live else 0
            except Exception:
                pass
            continue
        job_log = state_store.job_dir_from_id(spec.identity.run_id) / "job.log"
        if job_log.exists():
            totals["telemetry_jobs"] += 1
            totals["telemetry_dirs"] += 1
            batches, epochs = count_job_log_progress(job_log)
            totals["batches_done"] += batches
            totals["epochs_done"] += epochs
            if is_live:
                totals["live_batches_done"] += batches
                totals["live_epochs_done"] += epochs
            continue
        has_state_progress = state.get("history_rows") is not None or state.get("final_epoch") is not None
        if has_state_progress:
            totals["telemetry_jobs"] += 1
        for field, total_key, live_key in (
            ("history_rows", "batches_done", "live_batches_done"),
            ("final_epoch", "epochs_done", "live_epochs_done"),
        ):
            if state.get(field) is None:
                continue
            try:
                count = int(state[field] or 0)
            except Exception:
                continue
            totals[total_key] += count
            totals[live_key] += count if is_live else 0
    return totals
