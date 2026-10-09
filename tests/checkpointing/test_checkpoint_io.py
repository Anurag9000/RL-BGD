"""Atomic save preserves prior checkpoint and isolates concurrent staging files."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from typing import Any

import pytest
import torch

from rl_bgd.utils.checkpoint_io import save_training_checkpoint


def test_atomic_save_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "checkpoints" / "model.pt"
    save_training_checkpoint(target, {"value": torch.tensor([1.0])})
    save_training_checkpoint(target, {"value": torch.tensor([2.0])})
    torch.testing.assert_close(
        torch.load(target, weights_only=True)["value"],
        torch.tensor([2.0]),
    )
    assert list(target.parent.glob(".model.pt.*.tmp")) == []


def test_atomic_save_fsyncs_parent_directory_after_replace(
    tmp_path: Path, monkeypatch: Any
) -> None:
    target = tmp_path / "model.pt"
    original_fsync = os.fsync
    calls: list[int] = []

    def observed_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observed_fsync)
    save_training_checkpoint(target, {"value": torch.tensor([3.0])})

    expected_calls = 1 if os.name == "nt" else 2
    assert len(calls) == expected_calls


def test_failed_save_leaves_last_complete_checkpoint(tmp_path: Path, monkeypatch: Any) -> None:
    target = tmp_path / "model.pt"
    save_training_checkpoint(target, {"value": torch.tensor([7.0])})

    def fail_save(_state: Any, _handle: Any) -> None:
        raise OSError("simulated interrupted serialization")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interrupted serialization"):
        save_training_checkpoint(target, {"value": torch.tensor([99.0])})
    torch.testing.assert_close(
        torch.load(target, weights_only=True)["value"],
        torch.tensor([7.0]),
    )
    assert list(tmp_path.glob(".model.pt.*.tmp")) == []


def test_concurrent_writers_never_share_a_staging_file(
    tmp_path: Path, monkeypatch: Any
) -> None:
    target = tmp_path / "model.pt"
    original_save = torch.save
    barrier = Barrier(2)
    lock = Lock()
    paths: list[str] = []

    def observed_save(state: Any, handle: Any) -> None:
        with lock:
            paths.append(handle.name)
        barrier.wait(timeout=10)
        original_save(state, handle)

    monkeypatch.setattr(torch, "save", observed_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            pool.submit(save_training_checkpoint, target, {"value": torch.tensor([i])})
            for i in (1, 2)
        ]
        for result in results:
            result.result()

    assert len(set(paths)) == 2
    assert int(torch.load(target, weights_only=True)["value"].item()) in (1, 2)
    assert list(tmp_path.glob(".model.pt.*.tmp")) == []
