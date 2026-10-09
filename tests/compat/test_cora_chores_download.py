from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import Request

import pytest

from rl_bgd.compat import cora_chores_download as chores_download
from rl_bgd.compat.cora_chores_download import (
    ChoresArchiveDownloadError,
    download_chores_archive,
    extract_chores_archive,
)


class _FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        content_type: str,
    ) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = {"Content-Type": content_type}

    def read(
        self,
        amount: int = -1,
    ) -> bytes:
        return self._stream.read(amount)

    def __enter__(
        self,
    ) -> _FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None


def _zip_payload(
    members: dict[
        str,
        bytes,
    ],
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, payload in members.items():
            archive.writestr(
                name,
                payload,
            )
    return buffer.getvalue()


def test_download_falls_back_after_html_and_records_provenance(
    tmp_path: Path,
) -> None:
    archive_bytes = _zip_payload({"cora_trajs/train/demo/traj_data.json": (b"{}")})
    payloads = {
        "https://broken.example/archive": (
            b"<html>sign in</html>",
            "text/html",
        ),
        "https://mirror.example/archive": (
            archive_bytes,
            "application/zip",
        ),
    }

    def opener(
        request: Request,
        timeout: float,
    ) -> _FakeResponse:
        assert timeout == 12.0
        payload, content_type = payloads[request.full_url]
        return _FakeResponse(
            payload,
            content_type=content_type,
        )

    destination = tmp_path / "cora_trajs.zip"
    expected_sha = hashlib.sha256(archive_bytes).hexdigest()
    report = download_chores_archive(
        destination=destination,
        urls=(
            "https://broken.example/archive",
            "https://mirror.example/archive",
        ),
        expected_sha256=expected_sha,
        timeout=12.0,
        opener=opener,
    )

    assert destination.read_bytes() == (archive_bytes)
    assert report.source_url == ("https://mirror.example/archive")
    assert report.sha256 == expected_sha
    assert report.bytes_written == len(archive_bytes)
    assert len(report.candidate_failures) == 1
    assert "returned HTML" in (report.candidate_failures[0])


def test_download_retries_transient_network_failures(
    tmp_path: Path,
) -> None:
    archive_bytes = _zip_payload({"data/traj_data.json": b"{}"})
    attempts = 0

    def opener(
        request: Request,
        timeout: float,
    ) -> _FakeResponse:
        nonlocal attempts
        del request, timeout
        attempts += 1
        if attempts < 3:
            raise OSError("transient network failure")
        return _FakeResponse(
            archive_bytes,
            content_type="application/zip",
        )

    report = download_chores_archive(
        destination=(tmp_path / "archive.zip"),
        urls=("https://mirror.example/archive",),
        attempts_per_url=3,
        opener=opener,
    )

    assert attempts == 3
    assert len(report.candidate_failures) == 2
    assert all("network error" in failure for failure in (report.candidate_failures))


def test_download_rejects_non_https_candidate(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="must use HTTPS",
    ):
        download_chores_archive(
            destination=tmp_path / "archive.zip",
            urls=("http://mirror.example/archive",),
        )


def test_download_rejects_hash_mismatch(
    tmp_path: Path,
) -> None:
    archive_bytes = _zip_payload({"data/traj_data.json": b"{}"})

    def opener(
        request: Request,
        timeout: float,
    ) -> _FakeResponse:
        del request, timeout
        return _FakeResponse(
            archive_bytes,
            content_type="application/zip",
        )

    with pytest.raises(
        ChoresArchiveDownloadError,
        match="SHA-256 mismatch",
    ):
        download_chores_archive(
            destination=(tmp_path / "archive.zip"),
            urls=("https://mirror.example/archive",),
            expected_sha256=("0" * 64),
            opener=opener,
        )

    assert not (tmp_path / "archive.zip").exists()


def test_download_rejects_non_zip_payload(
    tmp_path: Path,
) -> None:
    def opener(
        request: Request,
        timeout: float,
    ) -> _FakeResponse:
        del request, timeout
        return _FakeResponse(
            b"not a zip",
            content_type=("application/octet-stream"),
        )

    with pytest.raises(
        ChoresArchiveDownloadError,
        match="not a ZIP archive",
    ):
        download_chores_archive(
            destination=(tmp_path / "archive.zip"),
            urls=("https://invalid.example/archive",),
            opener=opener,
        )


def test_non_zip_rejection_never_loads_entire_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed large downloads must not be read fully for error reporting."""

    payload = b"x" * 512

    def opener(request: Request, timeout: float) -> _FakeResponse:
        del request, timeout
        return _FakeResponse(payload, content_type="application/octet-stream")

    def forbid_read_bytes(self: Path) -> bytes:
        raise AssertionError("non-ZIP diagnostics must use bounded reads")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    with pytest.raises(ChoresArchiveDownloadError, match="not a ZIP archive"):
        download_chores_archive(
            destination=tmp_path / "invalid.zip",
            urls=("https://invalid.example/archive",),
            opener=opener,
            attempts_per_url=1,
        )


def test_download_attempts_use_distinct_temporary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate download attempts must never reuse a staging pathname."""

    destinations: list[Path] = []

    def reject_candidate(**kwargs: object) -> int:
        destination = kwargs["destination"]
        assert isinstance(destination, Path)
        destinations.append(destination)
        raise ChoresArchiveDownloadError("invalid candidate")

    monkeypatch.setattr(chores_download, "_download_candidate", reject_candidate)
    for _ in range(2):
        with pytest.raises(ChoresArchiveDownloadError, match="no CORA CHORES archive"):
            download_chores_archive(
                destination=tmp_path / "archive.zip",
                urls=("https://invalid.example/archive",),
                attempts_per_url=1,
            )

    assert len(destinations) == 2
    assert destinations[0] != destinations[1]
    assert all(not destination.exists() for destination in destinations)


def test_safe_extract_rejects_parent_traversal(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "unsafe.zip"
    archive.write_bytes(_zip_payload({"../escape.txt": b"bad"}))

    with pytest.raises(
        ChoresArchiveDownloadError,
        match="unsafe path",
    ):
        extract_chores_archive(
            archive=archive,
            destination=(tmp_path / "extract"),
        )

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_rejects_uncompressed_size_limit(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "oversized.zip"
    archive.write_bytes(
        _zip_payload(
            {
                "data/payload.bin": b"x" * 64,
            }
        )
    )

    destination = tmp_path / "extract"
    with pytest.raises(
        ChoresArchiveDownloadError,
        match="extraction size limit",
    ):
        extract_chores_archive(
            archive=archive,
            destination=destination,
            max_extracted_bytes=32,
        )

    assert not (destination / "data" / "payload.bin").exists()


def test_extract_failure_does_not_publish_partial_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "corrupt.zip"
    archive.write_bytes(_zip_payload({"a.txt": b"a", "b.txt": b"b"}))
    destination = tmp_path / "data"

    def fail_after_partial_write(
        self: zipfile.ZipFile,
        path: str | Path,
        members: object = None,
        pwd: object = None,
    ) -> None:
        del self, members, pwd
        (Path(path) / "partial.txt").write_bytes(b"incomplete")
        raise zipfile.BadZipFile("corrupted member CRC")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", fail_after_partial_write)
    with pytest.raises(zipfile.BadZipFile, match="corrupted member CRC"):
        extract_chores_archive(archive=archive, destination=destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".data.extract-*")) == []


def test_extract_rejects_nonempty_destination_without_overwriting(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    archive.write_bytes(_zip_payload({"subdir/entry.txt": b"new"}))
    destination = tmp_path / "data"
    destination.mkdir()
    original = destination / "existing.txt"
    original.write_bytes(b"untouched")

    with pytest.raises(ChoresArchiveDownloadError, match="not empty"):
        extract_chores_archive(archive=archive, destination=destination)

    assert original.read_bytes() == b"untouched"
    assert not (destination / "subdir").exists()


def test_extract_rejects_duplicate_member_paths(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("data/entry.txt", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            output.writestr("data/entry.txt", b"second")
    destination = tmp_path / "data"

    with pytest.raises(ChoresArchiveDownloadError, match="duplicate member path"):
        extract_chores_archive(archive=archive, destination=destination)

    assert not destination.exists()


def test_extract_accepts_empty_existing_destination(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    archive.write_bytes(_zip_payload({"data/file.txt": b"good"}))
    destination = tmp_path / "data"
    destination.mkdir()

    restored = extract_chores_archive(archive=archive, destination=destination)
    assert restored == destination.resolve()
    assert (destination / "data" / "file.txt").read_bytes() == b"good"


def test_safe_extract_accepts_regular_archive(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "safe.zip"
    archive.write_bytes(
        _zip_payload(
            {
                ("cora_trajs/train/demo/traj_data.json"): b"{}",
                ("cora_trajs/train/demo/raw_images/000.png"): b"png",
            }
        )
    )

    destination = tmp_path / "extract"
    resolved = extract_chores_archive(
        archive=archive,
        destination=destination,
    )
    assert resolved == (destination.resolve())
    assert (destination / "cora_trajs" / "train" / "demo" / "traj_data.json").is_file()
