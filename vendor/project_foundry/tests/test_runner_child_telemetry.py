import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from simple_adp.runner.telemetry import ChildTelemetryWriter, summarize_live


def _checkpoint(*, kind="active", epoch=0, cursor=0, widths=(4, 4)):
    return {
        "stage_status": kind,
        "hidden_widths": list(widths),
        "current_cursor": 1,
        "epoch": epoch,
        "batch_cursor": cursor,
        "batches_done": cursor,
        "patience": 3,
        "best_score": 0.25,
        "search_context": {
            "controller": "ADP-W-All",
            "phase": "width_search",
            "current_architecture": list(widths),
        },
    }


def test_child_telemetry_records_batches_epochs_and_one_final_state(tmp_path):
    writer = ChildTelemetryWriter(
        tmp_path / "result", job_id="run-1", model_id="ADP-W-All",
        checkpoint=tmp_path / "result" / "model.pt",
    )
    writer(_checkpoint(cursor=1))
    writer(_checkpoint(cursor=2))
    writer(_checkpoint(epoch=1, cursor=0))
    writer(_checkpoint(kind="completed", epoch=1, cursor=0))

    progress = json.loads(writer.progress_path.read_text())
    assert progress["history_rows"] == 2
    assert progress["final_epoch"] == 1
    assert progress["kind"] == "state"
    assert progress["architecture"] == [4, 4]
    with writer.csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["kind"] for row in rows] == [
        "batch", "batch", "epoch", "stage_state", "state", "stage_state",
    ]
    assert rows[0]["stage"] == "width_search"
    assert "checkpoint" not in rows[0]
    # Runner telemetry is orchestration-only and does not contain task metrics.
    assert "accuracy" not in rows[0] and "loss" not in rows[0]
    stage = json.loads(writer.stage_path.read_text())
    assert stage["schema"] == 2
    assert stage["completed"] is True
    assert stage["current_hidden_widths"] == [4, 4]
    assert stage["stage_checkpoint"].endswith("model.pt")
    rng_path = writer.results_dir / stage["rng_checkpoint_path"]
    assert rng_path.is_file()
    assert torch.load(rng_path, map_location="cpu", weights_only=False)["torch"].numel() > 0
    assert rows[-1]["kind"] == "stage_state"
    assert not {"rng_checkpoint", "architecture", "batch_cursor", "checkpoint", "active_stage"}.intersection(rows[-1])


def test_child_telemetry_checkpoint_writers_save_local_and_identical_mirrors(tmp_path):
    writer = ChildTelemetryWriter(
        tmp_path / "result", job_id="checkpoint-mirrors", model_id="ADP-W-All",
    )
    payload = {"weights": torch.arange(6, dtype=torch.float32), "epoch": 3}
    methods = (
        ("save_last_checkpoint", writer.last_checkpoint_path),
        ("save_best_checkpoint", writer.best_checkpoint_path),
        ("save_best_batch_checkpoint", writer.best_batch_checkpoint_path),
    )

    for name, local_path in methods:
        mirror_path = tmp_path / "stage" / local_path.name
        getattr(writer, name)(payload, mirror_path=mirror_path)
        assert local_path.is_file()
        assert mirror_path.is_file()
        assert local_path.read_bytes() == mirror_path.read_bytes()
        restored = torch.load(local_path, map_location="cpu", weights_only=False)
        assert restored["epoch"] == payload["epoch"]
        assert torch.equal(restored["weights"], payload["weights"])


def test_child_telemetry_resume_uses_progress_state_then_csv_fallback(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(result, job_id="run-2", model_id="MLP-Fixed")
    writer(_checkpoint(cursor=1))
    writer(_checkpoint(epoch=1, cursor=0))

    resumed = ChildTelemetryWriter(result, job_id="run-2", model_id="MLP-Fixed")
    assert (resumed.batches_done, resumed.epochs_done) == (1, 1)
    resumed.progress_path.write_text("not-json")
    fallback = ChildTelemetryWriter(result, job_id="run-2", model_id="MLP-Fixed")
    assert (fallback.batches_done, fallback.epochs_done) == (1, 1)


def test_child_counter_recovery_preserves_source_record_type_precedence(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(result, job_id="run-2b", model_id="MLP-Fixed")
    writer.csv_path.write_text(
        "kind,record_type\nepoch,batch\nfinal,epoch\n",
        encoding="utf-8",
    )
    writer.progress_path.write_bytes(b"\xff\xfe")

    recovered = ChildTelemetryWriter(result, job_id="run-2b", model_id="MLP-Fixed")

    # The child TrainingTelemetry loader prefers record_type and counts only
    # exact batch/epoch rows when progress_state.json cannot be read.
    assert (recovered.batches_done, recovered.epochs_done) == (1, 1)


def test_child_counter_recovery_preserves_negative_progress_values(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(result, job_id="run-2c", model_id="MLP-Fixed")
    writer.progress_path.write_text(
        json.dumps({"history_rows": -2, "final_epoch": -3}),
        encoding="utf-8",
    )

    recovered = ChildTelemetryWriter(result, job_id="run-2c", model_id="MLP-Fixed")

    assert (recovered.batches_done, recovered.epochs_done) == (-2, -3)


def test_child_stage_rng_sidecar_is_not_rewritten_for_each_batch(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(result, job_id="run-stage", model_id="ADP-W-All")
    writer(_checkpoint(cursor=1))
    assert not writer.stage_path.exists()
    assert not list(writer.root.glob("stage_state.rng.*.pt"))
    writer(_checkpoint(cursor=0, epoch=1))
    companions = list(writer.root.glob("stage_state.rng.*.pt"))
    assert len(companions) == 1
    stage = json.loads(writer.stage_path.read_text())
    assert stage["active_stage"] is True
    assert stage["epoch"] == 1


def test_child_telemetry_torch_replace_uses_fixed_temp_without_fsync(tmp_path, monkeypatch):
    import os
    import torch
    import simple_adp.runner.telemetry as telemetry_module

    target = tmp_path / "nested" / "state.pt"
    saved_to = []

    def save(payload, destination):
        saved_to.append(destination)
        destination.write_bytes(payload)

    def unexpected_fsync(_fd):
        raise AssertionError("telemetry Torch replacement must not fsync")

    monkeypatch.setattr(torch, "save", save)
    monkeypatch.setattr(os, "fsync", unexpected_fsync)
    ChildTelemetryWriter._atomic_torch_save(target, b"telemetry")

    assert saved_to == [target.with_suffix(".pt.tmp")]
    assert target.read_bytes() == b"telemetry"
    assert not saved_to[0].exists()


def test_child_stage_state_recovers_latest_matching_csv_row_when_json_is_missing(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(
        result, job_id="csv-stage-recovery", model_id="ADP-W-All",
    )
    writer(_checkpoint(kind="completed", epoch=2, cursor=0))
    writer.stage_path.unlink()

    recovered_writer = ChildTelemetryWriter(
        result, job_id="csv-stage-recovery", model_id="ADP-W-All",
    )
    # The copied loader treats CSV as a stage-state recovery source and then
    # requires its RNG companion. The current CSV schema omits that companion,
    # so this legacy fallback is intentionally an exact-resume failure.
    with pytest.raises(ValueError, match="exact-resume failure: stage RNG state cannot be restored"):
        recovered_writer.load_stage_state()


def test_child_stage_csv_fallback_appends_without_rewriting_legacy_header(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(result, job_id="legacy-csv-stage", model_id="MLP-Fixed")
    legacy_fields = (
        "timestamp", "kind", "stage", "split", "epoch", "batch", "batch_cursor",
        "patience", "architecture", "best_loss", "checkpoint", "active_stage",
        "artifact", "rng_checkpoint",
    )
    writer.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(legacy_fields)
    writer(_checkpoint(kind="completed", epoch=3, cursor=0))

    with writer.csv_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == list(legacy_fields)
    assert len(rows[1]) == len(legacy_fields)
    writer.stage_path.unlink()
    with pytest.raises(ValueError, match="exact-resume failure: stage RNG state cannot be restored"):
        writer.load_stage_state()


def test_default_stage_csv_omits_rng_companion_and_csv_recovery_cannot_restore_it(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(result, job_id="csv-rng-gap", model_id="ADP-W-All")
    writer(_checkpoint(epoch=1, cursor=0))

    with writer.csv_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert not {"rng_checkpoint", "architecture", "batch_cursor", "checkpoint", "active_stage"}.intersection(rows[-1])

    writer.stage_path.unlink()
    with pytest.raises(ValueError, match="exact-resume failure: stage RNG state cannot be restored"):
        writer.load_stage_state()


def test_legacy_stage_json_restores_rng_companion(tmp_path):
    import random
    import torch

    from simple_adp.checkpoint import capture_rng_state

    result = tmp_path / "result"
    writer = ChildTelemetryWriter(result, job_id="legacy-rng", model_id="ADP-W-All")
    rng_state = capture_rng_state()
    companion = writer.stage_path.parent / "legacy-stage-rng.pt"
    writer._atomic_torch_save(companion, rng_state)
    writer.stage_path.write_text(json.dumps({
        "kind": "stage_state", "rng_checkpoint": companion.name,
    }), encoding="utf-8")

    python_oracle = random.Random()
    python_oracle.setstate(rng_state["python"])
    expected_python = python_oracle.random()
    torch_oracle = torch.Generator().manual_seed(1)
    torch_oracle.set_state(rng_state["torch"])
    expected_torch = torch.rand(1, generator=torch_oracle)
    random.seed(999)
    torch.manual_seed(999)

    loaded = writer.load_stage_state()

    assert loaded["kind"] == "stage_state"
    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(1), expected_torch)
    assert writer._stage_rng_deferred_to_checkpoint is False


def test_active_stage_exact_estimator_checkpoint_owns_rng(tmp_path):
    import random
    import torch

    from simple_adp.checkpoint import CheckpointStore

    result = tmp_path / "result"
    checkpoint = result / "estimator.pt"
    writer = ChildTelemetryWriter(
        result, job_id="active-rng", model_id="ADP-W-All", checkpoint=checkpoint,
    )
    rng_state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": [],
    }
    CheckpointStore(checkpoint).save({
        "stage_state": {
            "model_state": {}, "optimizer_state": {}, "scheduler_state": {},
            "epoch": 1, "batch_cursor": 0, "rng_state": rng_state,
            "best_model_state": {}, "best_score": 0.0, "patience": 1,
        },
    }, signature="signature")
    writer.stage_path.write_text(json.dumps({
        "kind": "stage_state", "active_stage": True, "completed": False,
        "stage_checkpoint": str(checkpoint), "current_hidden_widths": [1],
        "rng_checkpoint_path": "missing-sidecar-rng.pt",
    }), encoding="utf-8")
    random.seed(731)
    expected_next = random.random()
    random.seed(731)

    loaded = writer.load_stage_state()

    assert loaded["active_stage"] is True
    assert writer._stage_rng_deferred_to_checkpoint is True
    assert random.random() == expected_next


def test_generic_stage_csv_fallback_metadata_contract():
    fields = set(ChildTelemetryWriter.CSV_FIELDS)
    omitted_state_fields = {
        "active_stage", "current_hidden_widths", "current_width_growth_cursor",
        "stage_checkpoint", "rng_checkpoint", "batch_cursor",
    }
    assert {"timestamp", "kind", "job_id", "model_id", "completed"} <= fields
    assert omitted_state_fields.isdisjoint(fields)




def test_child_stage_state_file_compatibility_reads_writes_and_wraps_raw_text(tmp_path):
    writer = ChildTelemetryWriter(
        tmp_path / "result", job_id="file-like-stage", model_id="ADP-W-All",
    )
    assert writer.exists() is False
    assert writer.read_text() == ""

    payload = {
        "stage_status": "completed",
        "hidden_widths": [3, 4],
        "current_cursor": 2,
        "epoch": 5,
        "batch_cursor": 1,
        "patience": 7,
        "best_score": 0.125,
        "search_context": {"controller": "ADP-W-All", "phase": "width_search"},
    }
    encoded = json.dumps(payload)
    assert writer.write_text(encoded) == len(encoded)
    assert writer.exists() is True
    stage_text = writer.read_text()
    assert stage_text == json.dumps(json.loads(stage_text), sort_keys=True, indent=2)
    stage = json.loads(stage_text)
    assert stage["current_hidden_widths"] == [3, 4]
    assert stage["current_width_growth_cursor"] == 2
    assert stage["controller"] == "ADP-W-All"
    assert json.loads(writer.progress_path.read_text())["kind"] == "stage_state"
    assert (writer.results_dir / stage["rng_checkpoint_path"]).is_file()

    raw = "legacy stage cursor"
    assert writer.write_text(raw) == len(raw)
    assert json.loads(writer.read_text())["raw_text"] == raw

    prior = writer.read_text()
    scalar = "17"
    assert writer.write_text(scalar) == len(scalar)
    assert writer.read_text() == prior


def _resume_payload(
    widths=(4, 4), *, phase="width_search", cursor=1,
    depth_bounds=None, controller="ADP-W-All",
):
    context = {
        "controller": controller,
        "phase": phase,
        "current_architecture": list(widths),
    }
    if depth_bounds is not None:
        context["depth_stage_bounds"] = depth_bounds
    state_dict = {"hidden.weight": torch.ones((widths[0], 2)), "hidden.bias": torch.ones(widths[0])}
    return {
        "search_context": context,
        "stage_state": {
            "stage_status": "active",
            "hidden_widths": list(widths),
            "current_cursor": cursor,
            "epoch": 1,
            "batch_cursor": 0,
            "batches_done": 0,
            "best_model_state": state_dict,
            "model_state": state_dict,
            "rng_state": {"python": (1,), "numpy": (1,), "torch": torch.get_rng_state()},
        },
    }


def test_active_stage_sidecar_matches_signature_checked_checkpoint(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(
        result, job_id="reconcile-1", model_id="ADP-W-All",
        checkpoint=result / "estimator.checkpoint.pt",
    )
    writer(_checkpoint(epoch=1, cursor=0))

    writer.reconcile_checkpoint(_resume_payload())

    stage = json.loads(writer.stage_path.read_text())
    assert stage["current_hidden_widths"] == [4, 4]
    assert "reconciled_from_checkpoint" not in stage


def test_active_stage_sidecar_allows_only_same_stage_monotonic_width_catchup(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(
        result, job_id="reconcile-2", model_id="ADP-W-All",
        checkpoint=result / "estimator.checkpoint.pt",
    )
    state = _checkpoint(epoch=1, cursor=0, widths=(3, 3))
    state["current_cursor"] = 2
    writer(state)

    writer.reconcile_checkpoint(_resume_payload((4, 4), cursor=2))

    stage = json.loads(writer.stage_path.read_text())
    assert stage["current_hidden_widths"] == [4, 4]
    assert stage["current_width_growth_cursor"] == 2
    assert stage["reconciled_from_checkpoint"] is True


def test_depth_stage_reconciliation_requires_matching_layer_and_width_bounds(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(
        result, job_id="depth-bounds", model_id="ADP-DW-D1-RR",
        checkpoint=result / "estimator.checkpoint.pt",
    )
    bounds = {"stage_index": 1, "start_width": 1, "target_width": 4}
    state = _checkpoint(epoch=1, cursor=0, widths=(3, 3))
    state["search_context"] = {
        "controller": "ADP-DW-D1-RR",
        "phase": "progressive_depth_step",
        "current_architecture": [3, 3],
        "depth_stage_bounds": bounds,
    }
    writer(state)

    writer.reconcile_checkpoint(_resume_payload(
        (3, 4), phase="progressive_depth_step", cursor=2, depth_bounds=bounds,
        controller="ADP-DW-D1-RR",
    ))

    sidecar = json.loads(writer.stage_path.read_text())
    assert sidecar["current_hidden_widths"] == [3, 4]
    assert sidecar["current_width_growth_cursor"] == 2
    assert sidecar["depth_stage_bounds"] == bounds
    assert sidecar["reconciled_from_checkpoint"] is True


@pytest.mark.parametrize(
    ("sidecar_widths", "checkpoint_widths", "bounds"),
    [
        # The source bounds every layer width, not only the inserted layer.
        ((5, 3), (5, 4), {"stage_index": 1, "start_width": 1, "target_width": 4}),
        # A depth-stage index is zero-based and must match architecture length.
        ((3, 3, 3), (3, 3, 4), {"stage_index": 1, "start_width": 1, "target_width": 4}),
    ],
)
def test_depth_stage_reconciliation_rejects_widths_outside_stage_shape_or_band(
    tmp_path, sidecar_widths, checkpoint_widths, bounds,
):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(
        result, job_id="depth-shape-invalid", model_id="ADP-DW-D1-RR",
        checkpoint=result / "estimator.checkpoint.pt",
    )
    state = _checkpoint(epoch=1, cursor=0, widths=sidecar_widths)
    state["search_context"] = {
        "controller": "ADP-DW-D1-RR",
        "phase": "progressive_depth_step",
        "current_architecture": list(sidecar_widths),
        "depth_stage_bounds": bounds,
    }
    writer(state)

    payload = _resume_payload(
        checkpoint_widths, phase="progressive_depth_step", cursor=2,
        depth_bounds=bounds, controller="ADP-DW-D1-RR",
    )
    with pytest.raises(ValueError, match="architecture conflicts"):
        writer.reconcile_checkpoint(payload)


@pytest.mark.parametrize(
    ("sidecar_bounds", "checkpoint_bounds", "checkpoint_widths"),
    [
        (None, {"stage_index": 1, "start_width": 1, "target_width": 4}, (3, 4)),
        ({"stage_index": 1, "start_width": 1, "target_width": 4}, {"stage_index": 0, "start_width": 1, "target_width": 4}, (4, 4)),
        ({"stage_index": 1, "start_width": 1, "target_width": 3}, {"stage_index": 1, "start_width": 1, "target_width": 3}, (3, 4)),
    ],
)
def test_depth_stage_reconciliation_rejects_missing_mismatched_or_out_of_bounds_metadata(
    tmp_path, sidecar_bounds, checkpoint_bounds, checkpoint_widths,
):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(
        result, job_id="depth-bounds-invalid", model_id="ADP-DW-D1-RR",
        checkpoint=result / "estimator.checkpoint.pt",
    )
    state = _checkpoint(epoch=1, cursor=0, widths=(3, 3))
    state["search_context"] = {
        "controller": "ADP-DW-D1-RR",
        "phase": "progressive_depth_step",
        "current_architecture": [3, 3],
        "depth_stage_bounds": sidecar_bounds,
    }
    writer(state)
    payload = _resume_payload(
        checkpoint_widths,
        phase="progressive_depth_step",
        cursor=2,
        depth_bounds=checkpoint_bounds,
        controller="ADP-DW-D1-RR",
    )

    with pytest.raises(ValueError, match="architecture conflicts"):
        writer.reconcile_checkpoint(payload)


@pytest.mark.parametrize(
    ("sidecar_widths", "checkpoint_widths", "phase"),
    [((4, 3), (3, 4), "width_search"), ((3, 3), (4, 4), "depth_search")],
)
def test_active_stage_sidecar_rejects_unrelated_architecture_mismatch(
    tmp_path, sidecar_widths, checkpoint_widths, phase,
):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(
        result, job_id="reconcile-invalid", model_id="ADP-W-All",
        checkpoint=result / "estimator.checkpoint.pt",
    )
    writer(_checkpoint(epoch=1, cursor=0, widths=sidecar_widths))

    with pytest.raises(ValueError, match="exact-resume failure"):
        writer.reconcile_checkpoint(_resume_payload(checkpoint_widths, phase=phase))


def test_stage_sidecar_reconciliation_fails_closed_on_corruption_and_path_escape(tmp_path):
    result = tmp_path / "result"
    writer = ChildTelemetryWriter(
        result, job_id="reconcile-corrupt", model_id="ADP-W-All",
        checkpoint=result / "estimator.checkpoint.pt",
    )
    writer(_checkpoint(epoch=1, cursor=0))
    writer.stage_path.write_text("not-json")
    with pytest.raises(ValueError, match="unreadable"):
        writer.reconcile_checkpoint(_resume_payload())

    writer(_checkpoint(epoch=1, cursor=0))
    stage = json.loads(writer.stage_path.read_text())
    stage["stage_checkpoint"] = str(tmp_path / "outside.pt")
    writer.stage_path.write_text(json.dumps(stage))
    with pytest.raises(ValueError, match="escapes the results directory"):
        writer.reconcile_checkpoint(_resume_payload())


def test_tabular_fit_reconciles_active_sidecar_before_exact_stage_resume(tmp_path):
    import numpy as np

    from simple_adp import CallableTask, SearchSettings, TabularADP

    calls = {"count": 0}

    def loss(outputs, targets):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("synthetic interruption after first batch")
        return torch.nn.functional.mse_loss(outputs, targets)

    task = CallableTask(
        1, loss,
        lambda outputs, targets: torch.nn.functional.mse_loss(outputs, targets),
        lambda outputs: outputs[:, 0],
        task_name="sidecar-resume",
    )
    settings = SearchSettings(
        init_width=2, init_depth=1, max_epochs=2, batch_size=2,
        max_width=2, max_depth=1, max_neurons=10,
        patience_width_exp=0, seed=31,
    )
    X = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    y = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    result = tmp_path / "result"
    checkpoint = result / "estimator.checkpoint.pt"
    mirrors = {
        "last": tmp_path / "jobs" / "resume-integration" / "checkpoint_last.pt",
        "best": tmp_path / "jobs" / "resume-integration" / "checkpoint_best.pt",
        "best_batch": tmp_path / "jobs" / "resume-integration" / "checkpoint_best_batch.pt",
    }
    writer = ChildTelemetryWriter(
        result, job_id="resume-integration", model_id="ADP-W-All", checkpoint=checkpoint,
        checkpoint_mirrors=mirrors,
    )
    interrupted = TabularADP("ADP-W-All", task=task, settings=settings)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        interrupted.fit(
            X[:4], y[:4], validation_data=(X[4:], y[4:]),
            checkpoint_path=checkpoint, progress_callback=writer,
        )

    sidecar_before = json.loads(writer.stage_path.read_text())
    assert sidecar_before["active_stage"] is True
    assert sidecar_before["current_hidden_widths"] == [2]
    assert writer.last_checkpoint_path.is_file()
    assert writer.best_checkpoint_path.is_file()
    assert writer.best_batch_checkpoint_path.is_file()
    assert mirrors["last"].read_bytes() == writer.last_checkpoint_path.read_bytes()
    assert mirrors["best"].read_bytes() == writer.best_checkpoint_path.read_bytes()
    assert mirrors["best_batch"].read_bytes() == writer.best_batch_checkpoint_path.read_bytes()
    initial_best_batch = torch.load(
        writer.best_batch_checkpoint_path, map_location="cpu", weights_only=False,
    )
    assert initial_best_batch["stage_checkpoint_kind"] == "initial_best_batch"
    assert initial_best_batch["best_batch_loss"] == float("inf")

    resumed_writer = ChildTelemetryWriter(
        result, job_id="resume-integration", model_id="ADP-W-All", checkpoint=checkpoint,
        checkpoint_mirrors=mirrors,
    )
    resumed = TabularADP("ADP-W-All", task=task, settings=settings)
    resumed.fit(
        X[:4], y[:4], validation_data=(X[4:], y[4:]),
        checkpoint_path=checkpoint, progress_callback=resumed_writer,
    )
    assert resumed.model_ is not None
    assert json.loads(resumed_writer.stage_path.read_text())["completed"] is True
    best_payload = torch.load(
        resumed_writer.best_checkpoint_path, map_location="cpu", weights_only=False,
    )
    best_stage = best_payload["stage_state"]
    assert best_stage["stage_status"] == "completed"
    assert best_stage["model_state"].keys() == best_stage["best_model_state"].keys()
    assert all(
        torch.equal(best_stage["model_state"][key], best_stage["best_model_state"][key])
        for key in best_stage["model_state"]
    )
    assert mirrors["best"].read_bytes() == resumed_writer.best_checkpoint_path.read_bytes()
    assert mirrors["last"].read_bytes() == resumed_writer.last_checkpoint_path.read_bytes()
    assert mirrors["best_batch"].read_bytes() == resumed_writer.best_batch_checkpoint_path.read_bytes()


def test_live_summary_counts_progress_sidecar_for_running_job(tmp_path):
    from types import SimpleNamespace

    result = tmp_path / "result"
    writer = ChildTelemetryWriter(result, job_id="run-3", model_id="ADP-D-D1")
    writer(_checkpoint(cursor=1))
    spec = SimpleNamespace(identity=SimpleNamespace(run_id="run-3"), result_dir=str(result))

    class Store:
        def load(self, _job_id):
            return {"status": "running"}

        def job_dir_from_id(self, job_id):
            return tmp_path / "jobs" / job_id

    summary = summarize_live(tmp_path, [spec], Store())
    assert summary["batches_done"] == summary["live_batches_done"] == 1
    assert summary["epochs_done"] == summary["live_epochs_done"] == 0
    assert summary["telemetry_jobs"] == summary["telemetry_dirs"] == 1


def test_live_summary_ignores_overflowing_persisted_progress_fields(tmp_path):
    from types import SimpleNamespace

    spec = SimpleNamespace(identity=SimpleNamespace(run_id="overflow-progress"), result_dir=str(tmp_path / "result"))

    class Store:
        def load(self, _job_id):
            return {"status": "completed", "history_rows": float("inf"), "final_epoch": 2}

        def job_dir_from_id(self, job_id):
            return tmp_path / "jobs" / job_id

    summary = summarize_live(tmp_path, [spec], Store())
    assert summary["batches_done"] == 0
    assert summary["epochs_done"] == 2
    assert summary["telemetry_jobs"] == 1


def test_live_summary_uses_newest_progress_sidecar_and_never_falls_back_when_malformed(tmp_path):
    from types import SimpleNamespace

    from simple_adp.runner.telemetry import summarize_live

    result = tmp_path / "results"
    telemetry_root = result / "telemetry"
    older = telemetry_root / "a" / "progress_state.json"
    newer = telemetry_root / "z" / "progress_state.json"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text(json.dumps({"history_rows": 8, "final_epoch": 4}), encoding="utf-8")
    newer.write_text(json.dumps({"history_rows": 3, "final_epoch": 2}), encoding="utf-8")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    spec = SimpleNamespace(identity=SimpleNamespace(run_id="newest"), result_dir=str(result))

    class Store:
        records = {"newest": {"status": "running"}}

        def load(self, job_id):
            return self.records.get(job_id)

        def job_dir_from_id(self, job_id):
            return tmp_path / "jobs" / job_id

    newest_summary = summarize_live(tmp_path, [spec], Store())
    assert (newest_summary["batches_done"], newest_summary["epochs_done"]) == (3, 2)
    assert (newest_summary["live_batches_done"], newest_summary["live_epochs_done"]) == (3, 2)

    # A present malformed sidecar suppresses both an existing log and persisted
    # progress fields, even when another (older) sidecar is valid.
    newer.write_text("{malformed", encoding="utf-8")
    Store.records["newest"].update(history_rows=99, final_epoch=99)
    job_log = Store().job_dir_from_id("newest") / "job.log"
    job_log.parent.mkdir(parents=True)
    job_log.write_text("batch=1/1 done\nepoch=1 val=0.1\n", encoding="utf-8")
    malformed_summary = summarize_live(tmp_path, [spec], Store())
    assert (malformed_summary["batches_done"], malformed_summary["epochs_done"]) == (0, 0)
    assert (malformed_summary["live_batches_done"], malformed_summary["live_epochs_done"]) == (0, 0)
    assert malformed_summary["telemetry_jobs"] == malformed_summary["telemetry_dirs"] == 1


def test_live_summary_existing_log_suppresses_persisted_progress_fallback(tmp_path):
    from types import SimpleNamespace

    from simple_adp.runner.telemetry import summarize_live

    spec = SimpleNamespace(identity=SimpleNamespace(run_id="log-precedence"), result_dir=str(tmp_path / "absent-result"))

    class Store:
        def load(self, _job_id):
            return {"status": "running", "history_rows": 20, "final_epoch": 5}

        def job_dir_from_id(self, job_id):
            return tmp_path / "jobs" / job_id

    log = Store().job_dir_from_id(spec.identity.run_id) / "job.log"
    log.parent.mkdir(parents=True)
    log.write_text("startup only; no terminal progress lines\n", encoding="utf-8")
    summary = summarize_live(tmp_path, [spec], Store())

    assert (summary["batches_done"], summary["epochs_done"]) == (0, 0)
    assert (summary["live_batches_done"], summary["live_epochs_done"]) == (0, 0)
    assert summary["telemetry_jobs"] == summary["telemetry_dirs"] == 1


def test_estimator_stage_callback_writes_worker_sidecars(tmp_path):
    import numpy as np

    from simple_adp import RegressionTask, SearchSettings, TabularADP

    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    writer = ChildTelemetryWriter(tmp_path, job_id="fit-1", model_id="ADP-W-All")
    estimator = TabularADP(
        "ADP-W-All",
        task=RegressionTask(),
        settings=SearchSettings(
            init_width=2, init_depth=1, max_epochs=1, batch_size=4,
            max_width=2, max_depth=1, max_neurons=10,
            patience_width_exp=0,
        ),
    )
    estimator.fit(X, y, validation_data=(X, y), progress_callback=writer)

    progress = json.loads(writer.progress_path.read_text())
    assert progress["history_rows"] == 1
    assert progress["final_epoch"] == 1
    assert progress["active_stage"] is False


def test_unreadable_stage_sidecar_is_reported_as_exact_resume_failure(tmp_path):
    writer = ChildTelemetryWriter(
        tmp_path / "result", job_id="run-corrupt-stage", model_id="ADP-W-All",
        checkpoint=tmp_path / "result" / "checkpoint.pt",
    )
    writer.stage_path.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="exact-resume failure: stage sidecar is unreadable"):
        writer.reconcile_checkpoint({})
