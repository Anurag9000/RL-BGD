"""Secure recovery helper for CORA's historical CHORES trajectory archive."""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

OFFICIAL_CHORES_ARCHIVE_URL = (
    "https://onedrive.live.com/download?"
    "cid=601D311D0FC404D4&"
    "resid=601D311D0FC404D4%2155915&"
    "authkey=APSnA-AKY4Yw_vA"
)
DEFAULT_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_EXTRACTED_BYTES = 16 * 1024 * 1024 * 1024


class _Response(Protocol):
    headers: object

    def read(
        self,
        amount: int = -1,
    ) -> bytes: ...

    def __enter__(self) -> "_Response": ...

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None: ...


OpenUrl = Callable[
    [Request, float],
    _Response,
]


class ChoresArchiveDownloadError(
    RuntimeError
):
    """Raised when no supplied authoritative archive candidate validates."""


@dataclass(frozen=True)
class ChoresArchiveDownloadReport:
    source_url: str
    destination: Path
    sha256: str
    bytes_written: int
    candidate_failures: tuple[
        str,
        ...,
    ]

    def to_dict(
        self,
    ) -> dict[str, object]:
        payload = asdict(
            self
        )
        payload[
            "destination"
        ] = str(
            self.destination
        )
        return payload


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1 << 20
            ),
            b"",
        ):
            digest.update(
                block
            )
    return digest.hexdigest()


def _response_content_type(
    response: _Response,
) -> str:
    headers = response.headers
    getter = getattr(
        headers,
        "get",
        None,
    )
    if getter is None:
        return ""
    value = getter(
        "Content-Type"
    )
    return (
        str(value).lower()
        if value is not None
        else ""
    )


def _download_candidate(
    *,
    url: str,
    destination: Path,
    timeout: float,
    max_bytes: int,
    opener: OpenUrl,
) -> int:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "RL-BGD-CORA-CHORES-Recovery/1"
            )
        },
    )
    bytes_written = 0
    with opener(
        request,
        timeout,
    ) as response:
        content_type = (
            _response_content_type(
                response
            )
        )
        if (
            "text/html"
            in content_type
        ):
            raise ChoresArchiveDownloadError(
                "remote source returned HTML instead of a trajectory archive"
            )

        with destination.open(
            "wb"
        ) as output:
            while True:
                chunk = response.read(
                    1 << 20
                )
                if not chunk:
                    break
                bytes_written += len(
                    chunk
                )
                if (
                    bytes_written
                    > max_bytes
                ):
                    raise ChoresArchiveDownloadError(
                        "remote archive exceeds configured size limit"
                    )
                output.write(
                    chunk
                )

    if bytes_written == 0:
        raise ChoresArchiveDownloadError(
            "remote source returned an empty file"
        )
    if not zipfile.is_zipfile(
        destination
    ):
        prefix = (
            destination.read_bytes()[
                :128
            ]
        )
        raise ChoresArchiveDownloadError(
            "downloaded payload is not a ZIP archive; "
            f"prefix={prefix!r}"
        )
    return bytes_written


def download_chores_archive(
    *,
    destination: str | Path,
    urls: Iterable[
        str
    ] = (
        OFFICIAL_CHORES_ARCHIVE_URL,
    ),
    expected_sha256: str | None = None,
    timeout: float = 60.0,
    max_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    attempts_per_url: int = 3,
    opener: OpenUrl | None = None,
) -> ChoresArchiveDownloadReport:
    """Download the first valid ZIP candidate and atomically publish it."""

    if timeout <= 0:
        raise ValueError(
            "timeout must be positive"
        )
    if max_bytes < 1:
        raise ValueError(
            "max_bytes must be positive"
        )

    if attempts_per_url < 1:
        raise ValueError(
            "attempts_per_url must be positive"
        )

    candidates = tuple(
        dict.fromkeys(
            url.strip()
            for url in urls
            if url.strip()
        )
    )
    if not candidates:
        raise ValueError(
            "at least one archive URL is required"
        )
    for url in candidates:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise ValueError(
                "archive URLs must use HTTPS with a network host"
            )

    expected = (
        expected_sha256.lower()
        if expected_sha256
        else None
    )
    if (
        expected is not None
        and (
            len(expected) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character in expected
            )
        )
    ):
        raise ValueError(
            "expected_sha256 must be a 64-character hexadecimal digest"
        )

    target = Path(
        destination
    )
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    open_url = (
        opener
        if opener is not None
        else lambda request, timeout_value: urlopen(  # noqa: S310
            request,
            timeout=timeout_value,
        )
    )

    failures: list[
        str
    ] = []
    for candidate_index, url in enumerate(
        candidates
    ):
        for attempt_index in range(
            attempts_per_url
        ):
            temporary = target.with_name(
                (
                    f".{target.name}.candidate-"
                    f"{candidate_index}-attempt-"
                    f"{attempt_index}.tmp"
                )
            )
            try:
                bytes_written = (
                    _download_candidate(
                        url=url,
                        destination=temporary,
                        timeout=timeout,
                        max_bytes=max_bytes,
                        opener=open_url,
                    )
                )
                digest = _sha256_file(
                    temporary
                )
                if (
                    expected is not None
                    and digest != expected
                ):
                    raise ChoresArchiveDownloadError(
                        "archive SHA-256 mismatch: "
                        f"expected {expected}, found {digest}"
                    )

                os.replace(
                    temporary,
                    target,
                )
                return (
                    ChoresArchiveDownloadReport(
                        source_url=url,
                        destination=(
                            target.resolve()
                        ),
                        sha256=digest,
                        bytes_written=(
                            bytes_written
                        ),
                        candidate_failures=tuple(
                            failures
                        ),
                    )
                )
            except OSError as exc:
                failures.append(
                    (
                        f"{url} attempt "
                        f"{attempt_index + 1}/"
                        f"{attempts_per_url}: "
                        f"network error: {exc}"
                    )
                )
                if (
                    attempt_index + 1
                    >= attempts_per_url
                ):
                    break
            except (
                ChoresArchiveDownloadError,
                zipfile.BadZipFile,
            ) as exc:
                failures.append(
                    f"{url}: {exc}"
                )
                break
            finally:
                temporary.unlink(
                    missing_ok=True
                )

    raise ChoresArchiveDownloadError(
        "no CORA CHORES archive candidate validated:\n"
        + "\n".join(
            failures
        )
    )


def extract_chores_archive(
    *,
    archive: str | Path,
    destination: str | Path,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
) -> Path:
    """Extract a validated ZIP with traversal, symlink, and size defenses."""

    if max_extracted_bytes < 1:
        raise ValueError(
            "max_extracted_bytes must be positive"
        )

    source = Path(
        archive
    )
    if not zipfile.is_zipfile(
        source
    ):
        raise ChoresArchiveDownloadError(
            f"not a ZIP archive: {source}"
        )

    target = Path(
        destination
    )
    target.mkdir(
        parents=True,
        exist_ok=True,
    )
    target_root = (
        target.resolve()
    )

    with zipfile.ZipFile(
        source
    ) as archive_file:
        members = archive_file.infolist()
        total_uncompressed = sum(
            member.file_size
            for member in members
            if not member.is_dir()
        )
        if total_uncompressed > max_extracted_bytes:
            raise ChoresArchiveDownloadError(
                "archive expands beyond configured extraction size limit"
            )

        for member in members:
            member_path = Path(
                member.filename
            )
            if (
                member_path.is_absolute()
                or ".."
                in member_path.parts
            ):
                raise ChoresArchiveDownloadError(
                    "archive contains an unsafe path: "
                    f"{member.filename}"
                )
            unix_mode = (
                member.external_attr
                >> 16
            )
            if (
                unix_mode
                & 0o170000
            ) == 0o120000:
                raise ChoresArchiveDownloadError(
                    "archive contains a symbolic link: "
                    f"{member.filename}"
                )
            resolved = (
                target
                / member_path
            ).resolve()
            if (
                resolved != target_root
                and target_root
                not in resolved.parents
            ):
                raise ChoresArchiveDownloadError(
                    "archive member escapes destination: "
                    f"{member.filename}"
                )

        archive_file.extractall(
            target
        )

    return target_root
