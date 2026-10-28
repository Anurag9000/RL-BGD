"""Durable per-candidate replay artifacts for generic ADP search."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


class GenericSearchArtifactStore:
    """Persist candidate, method, task, and replay records under one run root.

    The store is deliberately separate from ``CheckpointStore``: the latter
    owns exact in-stage resume, while this store retains completed candidate
    artifacts and permits a subsequent candidate to be built if the active
    stage checkpoint is unavailable.
    """

    SCHEMA = 2

    def __init__(self, root: str | os.PathLike[str], *, signature: str) -> None:
        self.root = Path(root)
        self.signature = str(signature)
        self.manifest_path = self.root / "manifest.json"
        self.candidates_root = self.root / "candidates"
        self.root.mkdir(parents=True, exist_ok=True)
        self._open_manifest()

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _atomic_torch(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _open_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"generic search manifest is unreadable: {exc}") from exc
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema") != self.SCHEMA
                or manifest.get("signature") != self.signature
            ):
                mismatch = {
                    "expected_signature": self.signature,
                    "found_signature": manifest.get("signature") if isinstance(manifest, dict) else None,
                    "expected_schema": self.SCHEMA,
                    "found_schema": manifest.get("schema") if isinstance(manifest, dict) else None,
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                }
                self._atomic_json(self.root / "manifest_mismatch.json", mismatch)
                raise ValueError("generic search artifact signature does not match this run")
            return
        existing = [path for path in self.root.iterdir() if path.name != "manifest_mismatch.json"]
        if existing:
            raise ValueError("generic search artifact directory has content but no signed manifest")
        self._atomic_json(self.manifest_path, {
            "schema": self.SCHEMA,
            "signature": self.signature,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    @staticmethod
    def _candidate_name(index: int) -> str:
        return f"candidate_{int(index):06d}"

    @staticmethod
    def _artifact_file(directory: Path, name: object) -> Path:
        """Resolve a checkpoint filename without following it outside its owner."""
        relative = Path(str(name))
        if relative.is_absolute() or relative.name != str(name) or str(name) in {"", ".", ".."}:
            raise ValueError("generic artifact checkpoint path must be a filename")
        root = directory.resolve()
        checkpoint = directory / relative
        try:
            checkpoint.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("generic artifact checkpoint path escapes its directory") from exc
        return checkpoint

    def record_candidate(
        self,
        *,
        index: int,
        phase: str,
        architecture: list[int],
        score: float,
        accepted: bool,
        context: dict[str, Any],
        candidate_best: dict[str, object],
        candidate_last: dict[str, object],
    ) -> str:
        name = self._candidate_name(index)
        directory = self.candidates_root / name
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_torch(directory / "checkpoint_best.pt", candidate_best)
        self._atomic_torch(directory / "checkpoint_last.pt", candidate_last)
        saved_context = dict(context)
        # The last candidate's best-epoch weights are the forward model used
        # as the warm start for the next proposal. Keep this separate from the
        # global-best snapshot in the search context.
        saved_context["artifact_current_snapshot"] = candidate_best
        saved_context.update({
            "schema": self.SCHEMA,
            "signature": self.signature,
            "artifact_candidate_id": name,
        })
        self._atomic_torch(directory / "search_context.pt", saved_context)
        record = {
            "schema": self.SCHEMA,
            "signature": self.signature,
            "candidate_id": name,
            "candidate_index": int(index),
            "phase": phase,
            "architecture": [int(width) for width in architecture],
            "score": float(score),
            "accepted": bool(accepted),
            "status": "completed",
            "best_checkpoint": "checkpoint_best.pt",
            "last_checkpoint": "checkpoint_last.pt",
            "context_checkpoint": "search_context.pt",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_json(directory / "candidate_state.json", record)
        return name

    def record_progress(
        self,
        *,
        candidate_id: str,
        context: dict[str, Any],
        candidate_score: float,
        candidate_accepted: bool,
        global_best: dict[str, object],
        task_complete: bool = False,
        task_score: float | None = None,
        task_history: list[dict[str, Any]] | None = None,
        task_state: dict[str, Any] | None = None,
    ) -> None:
        context_record = {
            "schema": self.SCHEMA,
            "signature": self.signature,
            "candidate_id": candidate_id,
            "phase": context.get("phase"),
            "profile": context.get("profile"),
            "candidate_score": float(candidate_score),
            "candidate_accepted": bool(candidate_accepted),
            "current_architecture": context.get("current_architecture"),
            "global_best_architecture": context.get("global_best_architecture"),
            "global_best_score": context.get("global_best_score"),
            "candidate_count": context.get("candidate_count", 0),
            "failure_counts": context.get("failure_counts", {}),
            "current_failures": context.get("current_failures", 0),
            "history": context.get("history", []),
            "status": "completed" if task_complete else "active",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        previous_best_dir = None
        previous_method_path = self.root / "method_state.json"
        if previous_method_path.exists():
            try:
                previous_method = json.loads(previous_method_path.read_text(encoding="utf-8"))
                if isinstance(previous_method, dict):
                    previous_best_dir = previous_method.get("best_candidate_dir")
            except Exception:
                previous_best_dir = None
        best_candidate_dir = (
            str((self.candidates_root / candidate_id).resolve())
            if candidate_accepted or not previous_best_dir else str(previous_best_dir)
        )
        method_record = dict(context_record)
        method_record.update({
            "best_val": float(candidate_score),
            "best_candidate_dir": best_candidate_dir,
        })
        task_record = dict(method_record)
        self._atomic_json(previous_method_path, method_record)
        self._atomic_json(self.root / "task_state.json", task_record)
        replay_path = self.root / "replay_manifest.json"
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {}
        except Exception:
            replay = {}
        entries = replay.get("candidates", []) if isinstance(replay, dict) else []
        if not isinstance(entries, list):
            entries = []
        entry = {
            "candidate_id": candidate_id,
            "phase": context.get("phase"),
            "score": float(candidate_score),
            "accepted": bool(candidate_accepted),
        }
        if not any(isinstance(item, dict) and item.get("candidate_id") == candidate_id for item in entries):
            entries.append(entry)
        self._atomic_json(self.root / "replay_manifest.json", {
            "schema": self.SCHEMA,
            "signature": self.signature,
            "status": "completed" if task_complete else "active",
            "candidates": entries,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        if task_complete:
            self._atomic_torch(self.root / "task_best.pt", global_best)
            method_record["status"] = "completed"
            self._atomic_json(self.root / "method_state.json", method_record)
            complete_record = dict(task_record)
            complete_record.update({
                "status": "completed",
                "task_score": float(task_score if task_score is not None else context.get("global_best_score", 0.0)),
                "history": task_history if task_history is not None else context.get("history", []),
                "search_state": task_state or {},
                "best_checkpoint": "task_best.pt",
            })
            self._atomic_json(self.root / "task_state.json", complete_record)

    def load_resume(self) -> dict[str, Any] | None:
        """Return a completed result or the latest completed candidate context."""
        task_path = self.root / "task_state.json"
        if task_path.exists():
            try:
                task_state = json.loads(task_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"generic task state is unreadable: {exc}") from exc
            if isinstance(task_state, dict) and task_state.get("signature") != self.signature:
                raise ValueError("generic task state signature does not match this run")
            if isinstance(task_state, dict) and task_state.get("schema") != self.SCHEMA:
                raise ValueError("generic task state schema does not match this implementation")
            if isinstance(task_state, dict) and task_state.get("status") == "completed":
                checkpoint_name = str(task_state.get("best_checkpoint", "task_best.pt"))
                checkpoint = self._artifact_file(self.root, checkpoint_name)
                snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)
                return {"completed": True, "state": task_state, "best_snapshot": snapshot}

        if not self.candidates_root.exists():
            return None
        candidates = sorted(self.candidates_root.glob("candidate_*/candidate_state.json"))
        for state_path in reversed(candidates):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if not isinstance(state, dict) or state.get("status") != "completed":
                    continue
                if state.get("signature") != self.signature:
                    raise ValueError("generic candidate signature does not match this run")
                if state.get("schema") != self.SCHEMA:
                    raise ValueError("generic candidate schema does not match this implementation")
                if state.get("candidate_id") != state_path.parent.name:
                    raise ValueError("generic candidate identity does not match its directory")
                if int(state.get("candidate_index", -1)) < 0 or self._candidate_name(
                    int(state["candidate_index"])
                ) != state_path.parent.name:
                    raise ValueError("generic candidate index does not match its directory")
                context_path = self._artifact_file(
                    state_path.parent, state.get("context_checkpoint", "")
                )
                context = torch.load(context_path, map_location="cpu", weights_only=False)
                if (
                    not isinstance(context, dict)
                    or context.get("schema") != self.SCHEMA
                    or context.get("signature") != self.signature
                    or context.get("artifact_candidate_id") != state.get("candidate_id")
                ):
                    raise ValueError("generic candidate context signature does not match this run")
                return {"completed": False, "state": state, "context": context}
            except Exception as exc:
                if any(marker in str(exc) for marker in (
                    "signature does not match", "path escapes", "must be a filename",
                    "identity does not match", "index does not match",
                )):
                    raise
                continue
        return None
