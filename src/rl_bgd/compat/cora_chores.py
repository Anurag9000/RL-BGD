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
                split: sum(ref.split == split for ref in self.trajectories)
                for split in sorted({ref.split for ref in self.trajectories})
            },
            "metadata_files": list(CHORES_METADATA_FILES),
        }


def _require_mapping(
    value: object,
    *,
    path: Path,
    label: str,
) -> Mapping[Any, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(f"{path} contains no {label} mapping")
    return value


def _require_sequence(
    value: object,
    *,
    path: Path,
    label: str,
) -> Sequence[Any]:
    if isinstance(
        value,
        (str, bytes),
    ) or not isinstance(
        value,
        Sequence,
    ):
        raise ValueError(f"{path} contains no usable {label} sequence")
    return value


def _require_nonempty_string(
    value: object,
    *,
    path: Path,
    label: str,
) -> str:
    if (
        not isinstance(
            value,
            str,
        )
        or not value
    ):
        raise ValueError(f"{path} contains no valid {label}")
    return value


def _validate_runtime_trajectory_contract(
    *,
    traj_path: Path,
    payload: Mapping[Any, Any],
) -> tuple[
    Sequence[Any],
    Sequence[Any],
    int,
]:
    """Validate fields consumed by the pinned crl_alfred reset/step path."""

    _require_nonempty_string(
        payload.get("task_type"),
        path=traj_path,
        label="task_type",
    )

    scene = _require_mapping(
        payload.get("scene"),
        path=traj_path,
        label="scene",
    )
    scene_num = scene.get("scene_num")
    if (
        isinstance(
            scene_num,
            bool,
        )
        or not isinstance(
            scene_num,
            int,
        )
        or scene_num < 1
    ):
        raise ValueError(f"{traj_path} has invalid scene.scene_num")
    _require_nonempty_string(
        scene.get("floor_plan"),
        path=traj_path,
        label="scene.floor_plan",
    )
    _require_sequence(
        scene.get("object_poses"),
        path=traj_path,
        label="scene.object_poses",
    )
    _require_sequence(
        scene.get("object_toggles"),
        path=traj_path,
        label="scene.object_toggles",
    )
    if not isinstance(
        scene.get("dirty_and_empty"),
        bool,
    ):
        raise ValueError(f"{traj_path} has invalid scene.dirty_and_empty")
    _require_mapping(
        scene.get("init_action"),
        path=traj_path,
        label="scene.init_action",
    )

    pddl = _require_mapping(
        payload.get("pddl_params"),
        path=traj_path,
        label="pddl_params",
    )
    for key in (
        "object_target",
        "parent_target",
        "toggle_target",
        "mrecep_target",
    ):
        if key not in pddl:
            raise ValueError(f"{traj_path} is missing pddl_params.{key}")
        value = pddl[key]
        if value is not None and not isinstance(
            value,
            str,
        ):
            raise ValueError(f"{traj_path} has invalid pddl_params.{key}")

    plan = _require_mapping(
        payload.get("plan"),
        path=traj_path,
        label="plan",
    )
    low_actions = _require_sequence(
        plan.get("low_actions"),
        path=traj_path,
        label="plan.low_actions",
    )
    if not low_actions:
        raise ValueError(f"{traj_path} contains no low_actions")
    for action_index, action in enumerate(low_actions):
        action_mapping = _require_mapping(
            action,
            path=traj_path,
            label=(f"plan.low_actions[{action_index}]"),
        )
        _require_mapping(
            action_mapping.get("api_action"),
            path=traj_path,
            label=(f"plan.low_actions[{action_index}].api_action"),
        )

    high_pddl = _require_sequence(
        plan.get("high_pddl"),
        path=traj_path,
        label="plan.high_pddl",
    )
    if not high_pddl:
        raise ValueError(f"{traj_path} contains no high_pddl")
    high_actions: list[str] = []
    for action_index, action in enumerate(high_pddl):
        action_mapping = _require_mapping(
            action,
            path=traj_path,
            label=(f"plan.high_pddl[{action_index}]"),
        )
        planner_action = _require_mapping(
            action_mapping.get("planner_action"),
            path=traj_path,
            label=(f"plan.high_pddl[{action_index}].planner_action"),
        )
        high_actions.append(
            _require_nonempty_string(
                planner_action.get("action"),
                path=traj_path,
                label=(f"plan.high_pddl[{action_index}].planner_action.action"),
            )
        )

    subgoal_count = len(high_actions)
    if high_actions[-1] == "End":
        subgoal_count -= 1
    if subgoal_count < 1:
        raise ValueError(f"{traj_path} has no executable high-level subgoals")

    return (
        low_actions,
        high_pddl,
        subgoal_count,
    )


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

    root = Path(metadata_root)
    refs: list[ChoresTrajectoryRef] = []
    for filename in CHORES_METADATA_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing CORA CHORES metadata file: {path}")
        payload = _load_json(path)
        if not isinstance(
            payload,
            list,
        ):
            raise TypeError(f"{path} must contain a list")
        for task_index, task in enumerate(payload):
            if not isinstance(
                task,
                Mapping,
            ):
                raise TypeError(f"{path} task {task_index} must be a mapping")
            task_name_raw = task.get("name")
            if (
                not isinstance(
                    task_name_raw,
                    str,
                )
                or not task_name_raw
            ):
                raise TypeError(f"{path} task {task_index} has no valid name")
            for split in (
                "train",
                "valid_seen",
            ):
                demos = task.get(split)
                if demos is None:
                    continue
                if isinstance(
                    demos,
                    (str, bytes),
                ) or not isinstance(
                    demos,
                    Sequence,
                ):
                    raise TypeError(
                        f"{path} task {task_name_raw!r} split {split!r} must be a sequence"
                    )
                for demo in demos:
                    if (
                        not isinstance(
                            demo,
                            str,
                        )
                        or not demo
                    ):
                        raise TypeError(
                            f"{path} task {task_name_raw!r} contains an invalid {split} demo"
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
    if len(unique) != len(refs):
        raise ValueError("CORA CHORES metadata contains duplicate split/demo references")
    if not refs:
        raise ValueError("CORA CHORES metadata references no trajectories")
    return tuple(refs)


def find_chores_archive_root(
    search_root: str | Path,
    refs: Sequence[ChoresTrajectoryRef],
) -> Path:
    """Locate the extracted archive root from a known referenced trajectory."""

    if not refs:
        raise ValueError("at least one trajectory reference is required")
    root = Path(search_root)
    first = refs[0]
    suffix = Path(first.split) / first.demo / "traj_data.json"
    matches = tuple(root.rglob(str(suffix)))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected exactly one extracted CORA CHORES archive root for "
            f"{suffix}, found {len(matches)}"
        )
    candidate = matches[0]
    for _ in suffix.parts:
        candidate = candidate.parent
    return candidate


def _validate_trajectory(
    *,
    archive_root: Path,
    ref: ChoresTrajectoryRef,
) -> int:
    demo_dir = archive_root / ref.split / ref.demo
    traj_path = demo_dir / "traj_data.json"
    if not traj_path.is_file():
        raise FileNotFoundError(f"missing CORA CHORES trajectory JSON: {traj_path}")
    payload = _load_json(traj_path)
    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(f"{traj_path} must contain a mapping")

    (
        low_actions,
        _,
        subgoal_count,
    ) = _validate_runtime_trajectory_contract(
        traj_path=traj_path,
        payload=payload,
    )
    images = _require_sequence(
        payload.get("images"),
        path=traj_path,
        label="images",
    )
    if not images:
        raise ValueError(f"{traj_path} contains no usable images")

    raw_images = demo_dir / "raw_images"
    if not raw_images.is_dir():
        raise FileNotFoundError(f"missing CORA CHORES raw_images directory: {raw_images}")

    checked = 0
    covered_high_indices: set[int] = set()
    for image_index, image in enumerate(images):
        if not isinstance(
            image,
            Mapping,
        ):
            raise TypeError(f"{traj_path} image {image_index} must be a mapping")
        image_name = image.get("image_name")
        low_idx = image.get("low_idx")
        high_idx = image.get("high_idx")
        if (
            not isinstance(
                image_name,
                str,
            )
            or not image_name
        ):
            raise TypeError(f"{traj_path} image {image_index} has no image_name")
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
            or low_idx >= len(low_actions)
        ):
            raise ValueError(f"{traj_path} image {image_index} has invalid low_idx")
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
            or high_idx >= subgoal_count
        ):
            raise ValueError(f"{traj_path} image {image_index} has invalid high_idx")
        covered_high_indices.add(high_idx)

        image_path = raw_images / (Path(image_name).stem + ".png")
        if not image_path.is_file():
            raise FileNotFoundError(f"missing CORA CHORES raw image: {image_path}")
        checked += 1

    expected_high_indices = set(range(subgoal_count))
    if covered_high_indices != expected_high_indices:
        missing = sorted(expected_high_indices - covered_high_indices)
        raise ValueError(
            f"{traj_path} images do not cover every executable high-level "
            f"subgoal; missing high_idx={missing}"
        )

    return checked


def validate_chores_archive(
    *,
    archive_root: str | Path,
    metadata_root: str | Path,
) -> ChoresArchiveReport:
    """Validate all trajectories and raw goal images referenced by CORA."""

    refs = load_chores_trajectory_refs(metadata_root)
    root = Path(archive_root)
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
        image_files_checked=(image_files_checked),
    )
