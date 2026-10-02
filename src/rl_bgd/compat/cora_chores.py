"""Validation helpers for the pinned CORA CHORES trajectory archive."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHORES_METADATA_FILES: tuple[str, ...] = (
    "multi_traj.json",
    "vary_envs.json",
    "vary_objects.json",
    "vary_tasks.json",
)
EXPECTED_CHORES_TRAJECTORY_COUNT = 27


@dataclass(frozen=True)
class ChoresTrajectoryRef:
    split: str
    demo: str
    source_file: str
    task_name: str


@dataclass(frozen=True)
class ChoresArchiveReport:
    archive_root: Path
    trajectories: tuple[ChoresTrajectoryRef, ...]
    image_files_checked: int

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "trajectory_count": len(self.trajectories),
            "image_files_checked": self.image_files_checked,
            "splits": {
                split: sum(
                    ref.split == split
                    for ref in self.trajectories
                )
                for split in sorted(
                    {
                        ref.split
                        for ref in self.trajectories
                    }
                )
            },
            "metadata_files": list(
                CHORES_METADATA_FILES
            ),
        }


def _load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_chores_trajectory_refs(
    metadata_root: str | Path,
) -> tuple[ChoresTrajectoryRef, ...]:
    """Load every train/valid_seen demo referenced by pinned CORA metadata."""

    root = Path(
        metadata_root
    )
    refs: list[
        ChoresTrajectoryRef
    ] = []
    for filename in CHORES_METADATA_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"missing CORA CHORES metadata file: {path}"
            )
        payload = _load_json(
            path
        )
        if not isinstance(
            payload,
            list,
        ):
            raise TypeError(
                f"{path} must contain a list"
            )
        for task_index, task in enumerate(
            payload
        ):
            if not isinstance(
                task,
                Mapping,
            ):
                raise TypeError(
                    f"{path} task {task_index} must be a mapping"
                )
            task_name_raw = task.get(
                "name"
            )
            if not isinstance(
                task_name_raw,
                str,
            ) or not task_name_raw:
                raise TypeError(
                    f"{path} task {task_index} has no valid name"
                )
            for split in (
                "train",
                "valid_seen",
            ):
                demos = task.get(
                    split
                )
                if demos is None:
                    continue
                if (
                    isinstance(
                        demos,
                        (str, bytes),
                    )
                    or not isinstance(
                        demos,
                        Sequence,
                    )
                ):
                    raise TypeError(
                        f"{path} task {task_name_raw!r} split {split!r} "
                        "must be a sequence"
                    )
                for demo in demos:
                    if not isinstance(
                        demo,
                        str,
                    ) or not demo:
                        raise TypeError(
                            f"{path} task {task_name_raw!r} contains "
                            f"an invalid {split} demo"
                        )
                    refs.append(
                        ChoresTrajectoryRef(
                            split=split,
                            demo=demo,
                            source_file=filename,
                            task_name=task_name_raw,
                        )
                    )

    unique = {
        (
            ref.split,
            ref.demo,
        )
        for ref in refs
    }
    if len(
        unique
    ) != len(
        refs
    ):
        raise ValueError(
            "CORA CHORES metadata contains duplicate split/demo references"
        )
    if not refs:
        raise ValueError(
            "CORA CHORES metadata references no trajectories"
        )
    return tuple(
        refs
    )


def find_chores_archive_root(
    search_root: str | Path,
    refs: Sequence[
        ChoresTrajectoryRef
    ],
) -> Path:
    """Locate the extracted archive root from a known referenced trajectory."""

    if not refs:
        raise ValueError(
            "at least one trajectory reference is required"
        )
    root = Path(
        search_root
    )
    first = refs[
        0
    ]
    suffix = (
        Path(
            first.split
        )
        / first.demo
        / "traj_data.json"
    )
    matches = tuple(
        root.rglob(
            str(
                suffix
            )
        )
    )
    if len(
        matches
    ) != 1:
        raise FileNotFoundError(
            "expected exactly one extracted CORA CHORES archive root for "
            f"{suffix}, found {len(matches)}"
        )
    candidate = matches[
        0
    ]
    for _ in suffix.parts:
        candidate = (
            candidate.parent
        )
    return candidate


def _validate_trajectory(
    *,
    archive_root: Path,
    ref: ChoresTrajectoryRef,
) -> int:
    demo_dir = (
        archive_root
        / ref.split
        / ref.demo
    )
    traj_path = (
        demo_dir
        / "traj_data.json"
    )
    if not traj_path.is_file():
        raise FileNotFoundError(
            f"missing CORA CHORES trajectory JSON: {traj_path}"
        )
    payload = _load_json(
        traj_path
    )
    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            f"{traj_path} must contain a mapping"
        )

    images = payload.get(
        "images"
    )
    if (
        isinstance(
            images,
            (str, bytes),
        )
        or not isinstance(
            images,
            Sequence,
        )
        or not images
    ):
        raise ValueError(
            f"{traj_path} contains no usable images"
        )

    plan = payload.get(
        "plan"
    )
    if not isinstance(
        plan,
        Mapping,
    ):
        raise ValueError(
            f"{traj_path} contains no plan mapping"
        )
    low_actions = plan.get(
        "low_actions"
    )
    if (
        isinstance(
            low_actions,
            (str, bytes),
        )
        or not isinstance(
            low_actions,
            Sequence,
        )
        or not low_actions
    ):
        raise ValueError(
            f"{traj_path} contains no low_actions"
        )

    raw_images = (
        demo_dir
        / "raw_images"
    )
    if not raw_images.is_dir():
        raise FileNotFoundError(
            f"missing CORA CHORES raw_images directory: {raw_images}"
        )

    checked = 0
    for image_index, image in enumerate(
        images
    ):
        if not isinstance(
            image,
            Mapping,
        ):
            raise TypeError(
                f"{traj_path} image {image_index} must be a mapping"
            )
        image_name = image.get(
            "image_name"
        )
        low_idx = image.get(
            "low_idx"
        )
        high_idx = image.get(
            "high_idx"
        )
        if not isinstance(
            image_name,
            str,
        ) or not image_name:
            raise TypeError(
                f"{traj_path} image {image_index} has no image_name"
            )
        if (
            isinstance(
                low_idx,
                bool,
            )
            or not isinstance(
                low_idx,
                int,
            )
            or low_idx < 0
            or low_idx
            >= len(
                low_actions
            )
        ):
            raise ValueError(
                f"{traj_path} image {image_index} has invalid low_idx"
            )
        if (
            isinstance(
                high_idx,
                bool,
            )
            or not isinstance(
                high_idx,
                int,
            )
            or high_idx < 0
        ):
            raise ValueError(
                f"{traj_path} image {image_index} has invalid high_idx"
            )

        image_path = (
            raw_images
            / (
                Path(
                    image_name
                ).stem
                + ".png"
            )
        )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"missing CORA CHORES raw image: {image_path}"
            )
        checked += 1

    return checked


def validate_chores_archive(
    *,
    archive_root: str | Path,
    metadata_root: str | Path,
) -> ChoresArchiveReport:
    """Validate all trajectories and raw goal images referenced by CORA."""

    refs = load_chores_trajectory_refs(
        metadata_root
    )
    root = Path(
        archive_root
    )
    image_files_checked = sum(
        _validate_trajectory(
            archive_root=root,
            ref=ref,
        )
        for ref in refs
    )
    return ChoresArchiveReport(
        archive_root=root,
        trajectories=refs,
        image_files_checked=(
            image_files_checked
        ),
    )
