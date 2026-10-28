import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_adp.runner.bundles import (
    PART_SIZE_BYTES,
    bundle_path_for_results_dir,
    export_results_bundle,
    rehydrate_sharded_telemetry,
    restore_results_bundle,
)
from simple_adp.runner.telemetry import summarize_offline


def test_bundle_path_uses_resolved_repo_relative_or_anchorless_external_parts(tmp_path):
    repository = tmp_path / "repo"
    inside = repository / "experiments" / "run one"
    outside = tmp_path / "external" / "results"
    inside.mkdir(parents=True)
    outside.mkdir(parents=True)
    alias = repository / "run-alias"
    alias.symlink_to(inside, target_is_directory=True)

    assert bundle_path_for_results_dir(alias, repository) == (
        repository / "resume_snapshots" / "experiments__run one.tar.gz"
    )
    outside_components = "__".join(outside.resolve().parts[1:])
    assert bundle_path_for_results_dir(outside, repository) == (
        repository / "resume_snapshots" / f"{outside_components}.tar.gz"
    )


def test_results_bundle_filters_telemetry_and_obeys_nonempty_restore_guard(tmp_path):
    repository = tmp_path / "repo"
    results = repository / "results"
    results.mkdir(parents=True)
    (results / "state.json").write_text("{}")
    (results / "telemetry.csv").write_text("kind\nbatch\n")
    (results / "nested").mkdir()
    (results / "nested" / "telemetry.csv").write_text("kind\nepoch\n")
    (results / "nested" / "not_telemetry.csv").write_text("keep\n")
    (results / "nested" / "telemetry.csv.manifest.json").write_text("{}")
    bundle = repository / "resume.tar.gz"

    assert PART_SIZE_BYTES == 64 * 1024 * 1024
    assert export_results_bundle(results, repository, bundle) == bundle
    manifest = json.loads(bundle.with_suffix("").with_suffix(".json").read_text())
    expected_source_size = sum(path.stat().st_size for path in results.rglob("*") if path.is_file())
    assert manifest["source_size_bytes"] == expected_source_size
    with tarfile.open(bundle, "r:gz") as archive:
        names = archive.getnames()
    assert "results/state.json" in names
    assert "results/telemetry.csv" not in names
    assert "results/nested/telemetry.csv" not in names
    assert "results/nested/not_telemetry.csv" in names
    assert "results/nested/telemetry.csv.manifest.json" in names
    assert not restore_results_bundle(results, repository, bundle)

    (results / "state.json").unlink()
    (results / "telemetry.csv").unlink()
    for path in (results / "nested").iterdir():
        path.unlink()
    (results / "nested").rmdir()
    results.rmdir()
    assert restore_results_bundle(results, repository, bundle)
    assert (results / "state.json").read_text() == "{}"
    assert not (results / "telemetry.csv").exists()
    assert not (results / "nested" / "telemetry.csv").exists()
    assert (results / "nested" / "not_telemetry.csv").read_text() == "keep\n"
    assert (results / "nested" / "telemetry.csv.manifest.json").read_text() == "{}"


def test_bundle_rejects_traversal_before_extracting(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    bundle = repository / "unsafe.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        info = tarfile.TarInfo("results/../escaped.txt")
        info.size = 3
        import io
        archive.addfile(info, io.BytesIO(b"bad"))

    with pytest.raises(ValueError, match="unsafe bundle member"):
        restore_results_bundle(results, repository, bundle)
    assert not (repository / "escaped.txt").exists()


def test_bundle_rejects_backslash_traversal_after_separator_normalization(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    bundle = repository / "unsafe-backslash.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        info = tarfile.TarInfo(r"results\..\escaped.txt")
        info.size = 3
        import io
        archive.addfile(info, io.BytesIO(b"bad"))

    with pytest.raises(ValueError, match="unsafe bundle member"):
        restore_results_bundle(results, repository, bundle)
    assert not (repository / "escaped.txt").exists()


@pytest.mark.parametrize(
    ("member_name", "member_type", "linkname", "error_match"),
    [
        ("/results/absolute.txt", tarfile.REGTYPE, "", "unsafe bundle member"),
        ("unexpected/file.txt", tarfile.REGTYPE, "", "unexpected bundle root"),
        ("results/symlink", tarfile.SYMTYPE, "../../outside.txt", "links are not permitted"),
        ("results/hardlink", tarfile.LNKTYPE, "results/state.json", "links are not permitted"),
    ],
)
def test_bundle_rejects_absolute_unexpected_root_and_link_members(
    tmp_path, member_name, member_type, linkname, error_match,
):
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    bundle = repository / "unsafe.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.type = member_type
        member.linkname = linkname
        if member_type == tarfile.REGTYPE:
            import io

            payload = b"must-not-extract"
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        else:
            archive.addfile(member)

    with pytest.raises(ValueError, match=error_match):
        restore_results_bundle(results, repository, bundle)
    assert not results.exists()
    assert not (repository / "outside.txt").exists()


def test_monolithic_restore_creates_missing_results_parent(tmp_path):
    repository = tmp_path / "repo"
    source = repository / "source" / "results"
    source.mkdir(parents=True)
    (source / "state.json").write_text("{}")
    bundle = repository / "snapshot.tar.gz"
    export_results_bundle(source, repository, bundle)

    target = repository / "not-created-yet" / "results"
    assert not target.parent.exists()
    assert restore_results_bundle(target, repository, bundle)
    assert (target / "state.json").read_text() == "{}"


def test_restore_nonempty_guard_returns_false_if_directory_listing_raises(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    results = repository / "results"
    results.mkdir(parents=True)
    original_iterdir = Path.iterdir

    def failing_iterdir(path):
        if path == results:
            raise RuntimeError("directory listing failed")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)
    assert restore_results_bundle(results, repository, repository / "missing.tar.gz") is False


def test_sharded_telemetry_rehydrates_with_checksum_and_enforces_root(tmp_path):
    repository = tmp_path / "repo"
    shard_dir = repository / "run" / "telemetry"
    shard_dir.mkdir(parents=True)
    target = repository / "run" / "telemetry.csv"
    shard = shard_dir / "telemetry.csv.part000"
    payload = b"kind,value\nbatch,1\n"
    shard.write_bytes(payload)
    manifest = shard_dir / "telemetry.csv.manifest.json"
    manifest.write_text(json.dumps({
        "original_path": str(target), "parts": [shard.name],
        "sha256": hashlib.sha256(payload).hexdigest(),
    }))
    assert rehydrate_sharded_telemetry(repository, repository) == 1
    assert target.read_bytes() == payload
    assert rehydrate_sharded_telemetry(repository, repository) == 0

    manifest.write_text(json.dumps({
        "original_path": str(tmp_path / "outside.csv"), "parts": [shard.name],
    }))
    with pytest.raises(ValueError, match="escapes repository"):
        rehydrate_sharded_telemetry(repository, repository)


def test_sharded_telemetry_relative_target_is_repo_anchored_and_existing_target_skips_parts(tmp_path):
    repository = tmp_path / "repo"
    shard_dir = repository / "run" / "nested" / "telemetry"
    shard_dir.mkdir(parents=True)
    payload = b"relative-target\n"
    (shard_dir / "data.part000").write_bytes(payload)
    manifest = shard_dir / "metrics.csv.manifest.json"
    manifest.write_text(json.dumps({
        # Relative original paths are rooted at repository_root, not beside the
        # manifest, matching the source helper's _manifest_target_path.
        "original_path": "run/metrics.csv",
        "parts": ["data.part000"],
    }))

    assert rehydrate_sharded_telemetry(repository, repository) == 1
    target = repository / "run" / "metrics.csv"
    assert target.read_bytes() == payload

    # Existing targets short-circuit before shard validation/opening.
    (shard_dir / "existing.csv.manifest.json").write_text(json.dumps({
        "original_path": "run/metrics.csv",
        "parts": ["missing.part000"],
    }))
    assert rehydrate_sharded_telemetry(repository, repository) == 0


def test_sharded_telemetry_malformed_missing_and_bad_checksum_residue(tmp_path):
    repository = tmp_path / "repo"
    shard_dir = repository / "run" / "telemetry"
    shard_dir.mkdir(parents=True)
    malformed = shard_dir / "broken.csv.manifest.json"
    malformed.write_text("{")
    assert rehydrate_sharded_telemetry(repository, repository) == 0

    missing_manifest = shard_dir / "missing.csv.manifest.json"
    missing_manifest.write_text(json.dumps({
        "original_path": str(repository / "run" / "missing.csv"),
        "parts": ["absent.part000"],
    }))
    with pytest.raises(FileNotFoundError):
        rehydrate_sharded_telemetry(repository, repository)
    missing_temp = repository / "run" / "missing.csv.rehydrate"
    assert missing_temp.exists()  # source leaves partial residue on missing-shard failure

    missing_manifest.unlink()
    payload = b"kind,value\nbatch,1\n"
    (shard_dir / "good.part000").write_bytes(payload)
    bad_manifest = shard_dir / "bad.csv.manifest.json"
    bad_manifest.write_text(json.dumps({
        "original_path": str(repository / "run" / "bad.csv"),
        "parts": ["good.part000"], "sha256": "0" * 64,
    }))
    with pytest.raises(ValueError, match="checksum mismatch"):
        rehydrate_sharded_telemetry(repository, repository)
    assert not (repository / "run" / "bad.csv.rehydrate").exists()


def test_sharded_telemetry_rejects_part_path_escape_before_reading_it(tmp_path):
    repository = tmp_path / "repo"
    shard_dir = repository / "run" / "telemetry"
    shard_dir.mkdir(parents=True)
    outside = repository / "outside.part000"
    outside.write_bytes(b"must-not-be-read")
    manifest = shard_dir / "escaped.csv.manifest.json"
    manifest.write_text(json.dumps({
        "original_path": str(repository / "run" / "escaped.csv"),
        "parts": ["../../outside.part000"],
    }))

    with pytest.raises(ValueError, match="shard escapes manifest directory"):
        rehydrate_sharded_telemetry(repository, repository)

    assert not (repository / "run" / "escaped.csv").exists()
    # The reference opens the rehydrate temporary before validating part paths.
    assert (repository / "run" / "escaped.csv.rehydrate").exists()


def test_results_bundle_chunks_and_restores_archive_using_part_order(tmp_path, monkeypatch):
    import simple_adp.runner.bundles as bundle_module

    monkeypatch.setattr(bundle_module, "PART_SIZE_BYTES", 128)
    repository = tmp_path / "repo"
    results = repository / "results"
    results.mkdir(parents=True)
    payload = bytes(range(256)) * 8
    (results / "payload.bin").write_bytes(payload)
    bundle = repository / "split.tar.gz"
    export_results_bundle(results, repository, bundle)
    assert not bundle.exists()
    parts = sorted(repository.glob("split.tar.gz.part*"))
    assert len(parts) > 1
    assert restore_results_bundle(results, repository, bundle) is False
    # The nonempty target guard applies even when a valid multipart archive exists.
    (results / "payload.bin").unlink()
    results.rmdir()
    assert restore_results_bundle(results, repository, bundle)
    assert (results / "payload.bin").read_bytes() == payload
    assert not bundle.with_suffix(bundle.suffix + ".restore").exists()


@pytest.mark.parametrize(("archive_size", "expected_parts"), [(128, []), (129, [128, 1])])
def test_bundle_export_uses_inclusive_part_size_boundary_and_removes_stale_files(
    tmp_path, monkeypatch, archive_size, expected_parts,
):
    import simple_adp.runner.bundles as bundle_module

    monkeypatch.setattr(bundle_module, "PART_SIZE_BYTES", 128)
    repository = tmp_path / "repo"
    results = repository / "results"
    results.mkdir(parents=True)
    (results / "state.json").write_text("{}")
    bundle = repository / "snapshot.tar.gz"
    bundle.write_bytes(b"old monolithic archive")
    stale_part = repository / "snapshot.tar.gz.part009"
    stale_part.write_bytes(b"old part")
    payload = bytes((index % 251 for index in range(archive_size)))
    calls = {}

    class FakeArchive:
        def __init__(self, path, mode, compresslevel):
            calls.update(mode=mode, compresslevel=compresslevel)
            self.stream = Path(path).open("wb")

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.stream.close()

        def add(self, *_args, **_kwargs):
            self.stream.write(payload)

    monkeypatch.setattr(bundle_module.tarfile, "open", FakeArchive)
    assert export_results_bundle(results, repository, bundle) == bundle

    parts = sorted(repository.glob("snapshot.tar.gz.part*"))
    assert [part.stat().st_size for part in parts] == expected_parts
    assert stale_part not in parts
    assert calls == {"mode": "w:gz", "compresslevel": 1}
    if archive_size == 128:
        assert bundle.read_bytes() == payload
    else:
        assert not bundle.exists()
        assert b"".join(part.read_bytes() for part in parts) == payload
    assert not bundle.with_suffix(bundle.suffix + ".tmp").exists()


def test_multipart_restore_creates_target_parent_before_reading_parts(tmp_path, monkeypatch):
    import simple_adp.runner.bundles as bundle_module

    monkeypatch.setattr(bundle_module, "PART_SIZE_BYTES", 128)
    repository = tmp_path / "repo"
    source = repository / "source" / "results"
    source.mkdir(parents=True)
    (source / "payload.bin").write_bytes(bytes(range(256)) * 8)
    bundle = repository / "snapshots" / "resume.tar.gz"
    export_results_bundle(source, repository, bundle)
    parts = set(bundle.parent.glob(f"{bundle.name}.part*"))
    assert parts

    target = repository / "new-parent" / "results"
    original_open = Path.open
    parent_presence_at_part_read = []

    def observed_open(path, mode="r", *args, **kwargs):
        if path in parts and mode == "rb":
            parent_presence_at_part_read.append(target.parent.exists())
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    assert restore_results_bundle(target, repository, bundle)
    assert parent_presence_at_part_read and all(parent_presence_at_part_read)
    assert (target / "payload.bin").read_bytes() == bytes(range(256)) * 8


def test_multipart_restore_keeps_reassembly_file_when_archive_open_fails(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    bundle = repository / "broken.tar.gz"
    part = repository / "broken.tar.gz.part000"
    part.write_bytes(b"not a gzip tar archive")

    with pytest.raises(tarfile.ReadError):
        restore_results_bundle(results, repository, bundle)

    reassembled = bundle.with_suffix(bundle.suffix + ".restore")
    assert reassembled.read_bytes() == part.read_bytes()


def test_offline_telemetry_prefers_csv_over_jsonl_and_counts_only_batch_epoch(tmp_path):
    result = tmp_path / "job"
    telemetry = result / "telemetry"
    telemetry.mkdir(parents=True)
    (telemetry / "telemetry.csv").write_text(
        "kind,record_type\nbatch,epoch\nepoch,batch\nfinal,\nother,\n"
    )
    (telemetry / "batch_metrics.jsonl").write_text("{}\n{}\n{}\n")
    spec = SimpleNamespace(result_dir=str(result))
    assert summarize_offline([spec]) == {
        "batches_done": 1, "epochs_done": 1,
        "telemetry_jobs": 1, "telemetry_dirs": 1,
    }
