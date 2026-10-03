from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rl_bgd.compat.cora_chores import (
    CHORES_METADATA_FILES,
    find_chores_archive_root,
    load_chores_trajectory_refs,
    validate_chores_archive,
)


def _write_json(
    path: Path,
    payload: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _fixture_tree(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    str,
]:
    metadata_root = tmp_path / "metadata"
    demo = (
        "pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-402/trial_T20210817_071626_357261"
    )
    _write_json(
        metadata_root / "vary_tasks.json",
        [
            {
                "name": ("hang_toilet_paper"),
                "train": [demo],
            }
        ],
    )
    for filename in set(CHORES_METADATA_FILES) - {"vary_tasks.json"}:
        _write_json(
            metadata_root / filename,
            [],
        )

    archive_root = tmp_path / "extracted" / "cora_trajs"
    demo_root = archive_root / "train" / demo
    _write_json(
        demo_root / "traj_data.json",
        {
            "images": [
                {
                    "image_name": ("000000000.png"),
                    "low_idx": 0,
                    "high_idx": 0,
                }
            ],
            "plan": {"low_actions": [{"api_action": {}}]},
        },
    )
    raw_images = demo_root / "raw_images"
    raw_images.mkdir(
        parents=True,
        exist_ok=True,
    )
    (raw_images / "000000000.png").write_bytes(b"synthetic")
    return (
        metadata_root,
        archive_root,
        demo,
    )


def test_chores_archive_discovery_and_validation(
    tmp_path: Path,
) -> None:
    (
        metadata_root,
        archive_root,
        demo,
    ) = _fixture_tree(tmp_path)
    refs = load_chores_trajectory_refs(metadata_root)
    assert len(refs) == 1
    assert refs[0].demo == demo

    discovered = find_chores_archive_root(
        tmp_path / "extracted",
        refs,
    )
    assert discovered == archive_root

    report = validate_chores_archive(
        archive_root=discovered,
        metadata_root=metadata_root,
    )
    assert report.archive_root == archive_root
    assert len(report.trajectories) == 1
    assert report.image_files_checked == 1
    payload = report.to_dict()
    assert payload["trajectory_count"] == 1
    assert payload["splits"] == {"train": 1}


def test_chores_archive_rejects_missing_raw_image(
    tmp_path: Path,
) -> None:
    (
        metadata_root,
        archive_root,
        demo,
    ) = _fixture_tree(tmp_path)
    (archive_root / "train" / demo / "raw_images" / "000000000.png").unlink()

    with pytest.raises(
        FileNotFoundError,
        match="raw image",
    ):
        validate_chores_archive(
            archive_root=archive_root,
            metadata_root=metadata_root,
        )


def test_chores_archive_rejects_missing_referenced_trajectory(
    tmp_path: Path,
) -> None:
    (
        metadata_root,
        archive_root,
        demo,
    ) = _fixture_tree(tmp_path)
    (archive_root / "train" / demo / "traj_data.json").unlink()

    with pytest.raises(
        FileNotFoundError,
        match="trajectory JSON",
    ):
        validate_chores_archive(
            archive_root=archive_root,
            metadata_root=metadata_root,
        )


def test_chores_archive_root_requires_unique_match(
    tmp_path: Path,
) -> None:
    (
        metadata_root,
        archive_root,
        demo,
    ) = _fixture_tree(tmp_path)
    refs = load_chores_trajectory_refs(metadata_root)
    duplicate = tmp_path / "extracted" / "duplicate" / "train" / demo
    duplicate.mkdir(
        parents=True,
        exist_ok=True,
    )
    (duplicate / "traj_data.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="exactly one",
    ):
        find_chores_archive_root(
            tmp_path / "extracted",
            refs,
        )

    assert archive_root.is_dir()


def test_chores_archive_cli_root_only(tmp_path: Path) -> None:
    metadata_root, archive_root, _ = _fixture_tree(tmp_path)
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "validate_cora_chores_archive.py"),
            "--search-root",
            str(tmp_path / "extracted"),
            "--metadata-root",
            str(metadata_root),
            "--expected-trajectories",
            "1",
            "--root-only",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(completed.stdout.strip()) == archive_root


def test_chores_archive_cli_defaults_to_pinned_cardinality(tmp_path: Path) -> None:
    metadata_root, _, _ = _fixture_tree(tmp_path)
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "validate_cora_chores_archive.py"),
            "--search-root",
            str(tmp_path / "extracted"),
            "--metadata-root",
            str(metadata_root),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "expected 27, found 1" in completed.stderr
