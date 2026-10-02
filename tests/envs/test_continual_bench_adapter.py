from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from rl_bgd.envs.continual_bench import (
    ContinualBenchStreamConfig,
    ContinualBenchStreamEnv,
)
from rl_bgd.envs.continual_bench.stream import (
    _install_pinned_runtime_compatibility,
    _repair_missing_metaworld_assets,
)


class FakeBox:
    def __init__(
        self,
        low: list[float],
        high: list[float],
    ) -> None:
        self.low = np.asarray(
            low,
            dtype=np.float32,
        )
        self.high = np.asarray(
            high,
            dtype=np.float32,
        )
        self.shape = self.low.shape


class FakeContinualBench:
    all_tasks = [
        "button",
        "door",
        "window",
        "faucet",
        "peg",
        "block",
    ]

    def __init__(
        self,
        *,
        five_value: bool = False,
        success_on_step: int | None = None,
    ) -> None:
        self.action_space = FakeBox(
            [-1.0],
            [1.0],
        )
        self.observation_space = FakeBox(
            [-10.0, -10.0],
            [10.0, 10.0],
        )
        self.task = "button"
        self.steps = 0
        self.five_value = five_value
        self.success_on_step = success_on_step

    def set_task(
        self,
        task: str,
    ) -> None:
        self.task = task

    def reset(
        self,
        seed: int | None = None,
    ) -> np.ndarray:
        del seed
        return np.asarray(
            [
                float(self.all_tasks.index(self.task)),
                0.0,
            ],
            dtype=np.float32,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Any, ...]:
        assert action.shape == (1,)
        self.steps += 1
        success = self.success_on_step is not None and self.steps == self.success_on_step
        observation = np.asarray(
            [
                float(self.all_tasks.index(self.task)),
                float(self.steps),
            ],
            dtype=np.float32,
        )
        rewards = {name: float(index + 1) for index, name in enumerate(self.all_tasks)}
        info = {
            name: {"success": (success if name == self.task else False)} for name in self.all_tasks
        }
        if self.five_value:
            return (
                observation,
                rewards,
                False,
                False,
                info,
            )
        return (
            observation,
            rewards,
            False,
            info,
        )


def test_fixed_step_switch_hides_task_and_selects_active_reward() -> None:
    base = FakeContinualBench()
    env = ContinualBenchStreamEnv(
        base,
        config=ContinualBenchStreamConfig(
            task_sequence=(
                "button",
                "door",
            ),
            switch_mode="fixed_steps",
            steps_per_task=2,
        ),
    )
    observation, info = env.reset(seed=0)
    assert observation.shape == (2,)
    assert info == {}

    _, reward, terminated, truncated, info = env.step(torch.tensor([0.0]))
    assert reward == pytest.approx(1.0)
    assert not terminated
    assert not truncated
    assert info == {}

    observation, reward, _, _, info = env.step(torch.tensor([0.0]))
    assert reward == pytest.approx(1.0)
    assert info == {}
    assert env.evaluation_context["task_name"] == "door"
    assert observation[0].item() == pytest.approx(1.0)


def test_success_switch_supports_five_value_api() -> None:
    env = ContinualBenchStreamEnv(
        FakeContinualBench(
            five_value=True,
            success_on_step=1,
        ),
        config=ContinualBenchStreamConfig(
            task_sequence=(
                "button",
                "door",
            ),
            switch_mode="success",
        ),
    )
    env.reset()
    observation, reward, terminated, truncated, info = env.step(torch.tensor([0.0]))
    assert reward == pytest.approx(1.0)
    assert not terminated
    assert not truncated
    assert info == {}
    assert env.evaluation_context["task_name"] == "door"
    assert observation[0].item() == pytest.approx(1.0)


def test_missing_visual_assets_are_repaired_from_metaworld(
    tmp_path: Path,
) -> None:
    continual_package = tmp_path / "continual_bench" / "envs"
    metaworld_package = tmp_path / "metaworld"
    destination_root = continual_package / "assets"
    source_root = metaworld_package / "assets"

    dependency_xml = destination_root / "objects" / "assets" / "buttonbox_dependencies.xml"
    dependency_xml.parent.mkdir(parents=True)
    dependency_xml.write_text(
        '<mujocoinclude><asset><texture file="../textures/metal1.png"/></asset></mujocoinclude>',
        encoding="utf-8",
    )

    canonical_texture = source_root / "textures" / "metal1.png"
    canonical_texture.parent.mkdir(parents=True)
    canonical_texture.write_bytes(b"canonical-metal")

    repaired = _repair_missing_metaworld_assets(
        SimpleNamespace(__file__=str(continual_package / "__init__.py")),
        SimpleNamespace(__file__=str(metaworld_package / "__init__.py")),
    )

    restored = destination_root / "objects" / "textures" / "metal1.png"
    assert repaired == ("objects/textures/metal1.png",)
    assert restored.read_bytes() == b"canonical-metal"


def test_asset_repair_never_overwrites_existing_benchmark_file(
    tmp_path: Path,
) -> None:
    continual_package = tmp_path / "continual_bench" / "envs"
    metaworld_package = tmp_path / "metaworld"
    destination_root = continual_package / "assets"
    source_root = metaworld_package / "assets"

    root_xml = destination_root / "root.xml"
    root_xml.parent.mkdir(parents=True)
    root_xml.write_text(
        '<mujoco><asset><texture file="objects/textures/metal1.png"/></asset></mujoco>',
        encoding="utf-8",
    )

    existing = destination_root / "objects" / "textures" / "metal1.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"benchmark-version")

    canonical = source_root / "textures" / "metal1.png"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"metaworld-version")

    repaired = _repair_missing_metaworld_assets(
        SimpleNamespace(__file__=str(continual_package / "__init__.py")),
        SimpleNamespace(__file__=str(metaworld_package / "__init__.py")),
    )

    assert repaired == ()
    assert existing.read_bytes() == b"benchmark-version"


def test_asset_repair_collapses_duplicated_legacy_root(
    tmp_path: Path,
) -> None:
    continual_package = tmp_path / "continual_bench" / "envs"
    metaworld_package = tmp_path / "metaworld"
    destination_root = continual_package / "assets"
    source_root = metaworld_package / "assets"

    dependency_xml = destination_root / "objects" / "assets" / "assembly_peg_dependencies.xml"
    dependency_xml.parent.mkdir(parents=True)
    dependency_xml.write_text(
        '<mujoco><asset><mesh file="../objects/meshes/assembly_peg/handle.stl"/></asset></mujoco>',
        encoding="utf-8",
    )

    canonical = source_root / "objects" / "meshes" / "assembly_peg" / "handle.stl"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical-mesh")

    repaired = _repair_missing_metaworld_assets(
        SimpleNamespace(__file__=str(continual_package / "__init__.py")),
        SimpleNamespace(__file__=str(metaworld_package / "__init__.py")),
    )

    restored = destination_root / "objects" / "objects" / "meshes" / "assembly_peg" / "handle.stl"
    assert repaired == ("objects/objects/meshes/assembly_peg/handle.stl",)
    assert restored.read_bytes() == b"canonical-mesh"


def test_asset_repair_ignores_unreachable_stale_xml(
    tmp_path: Path,
) -> None:
    continual_package = tmp_path / "continual_bench" / "envs"
    metaworld_package = tmp_path / "metaworld"
    destination_root = continual_package / "assets"
    source_root = metaworld_package / "assets"

    root_xml = destination_root / "sawyer_xyz" / "sawyer_bench.xml"
    root_xml.parent.mkdir(parents=True)
    root_xml.write_text(
        '<mujoco><include file="../objects/assets/buttonbox_dependencies.xml"/></mujoco>',
        encoding="utf-8",
    )

    dependency = destination_root / "objects" / "assets" / "buttonbox_dependencies.xml"
    dependency.parent.mkdir(parents=True)
    dependency.write_text(
        '<mujoco><asset><texture file="../textures/metal1.png"/></asset></mujoco>',
        encoding="utf-8",
    )

    stale = destination_root / "objects" / "assets" / "xyz_base.xml"
    stale.write_text(
        '<mujoco><include file="shared_config.xml"/></mujoco>',
        encoding="utf-8",
    )

    canonical = source_root / "textures" / "metal1.png"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical-metal")

    repaired = _repair_missing_metaworld_assets(
        SimpleNamespace(__file__=str(continual_package / "__init__.py")),
        SimpleNamespace(__file__=str(metaworld_package / "__init__.py")),
    )

    assert repaired == ("objects/textures/metal1.png",)
    assert not (dependency.parent / "shared_config.xml").exists()


def test_asset_repair_ignores_commented_file_references(
    tmp_path: Path,
) -> None:
    continual_package = tmp_path / "continual_bench" / "envs"
    metaworld_package = tmp_path / "metaworld"
    destination_root = continual_package / "assets"
    source_root = metaworld_package / "assets"

    root_xml = destination_root / "sawyer_xyz" / "sawyer_bench.xml"
    root_xml.parent.mkdir(parents=True)
    root_xml.write_text(
        '<mujoco><include file="../objects/assets/xyz_base.xml"/></mujoco>',
        encoding="utf-8",
    )

    xyz_base = destination_root / "objects" / "assets" / "xyz_base.xml"
    xyz_base.parent.mkdir(parents=True)
    xyz_base.write_text(
        (
            "<mujocoinclude>"
            '<!-- <include file="shared_config.xml"/> -->'
            '<geom type="box" size="1 1 1"/>'
            "</mujocoinclude>"
        ),
        encoding="utf-8",
    )

    source_root.mkdir(parents=True)

    repaired = _repair_missing_metaworld_assets(
        SimpleNamespace(__file__=str(continual_package / "__init__.py")),
        SimpleNamespace(__file__=str(metaworld_package / "__init__.py")),
    )

    assert repaired == ()
    assert not (xyz_base.parent / "shared_config.xml").exists()


class FakeUnclosableContinualBench(FakeContinualBench):
    def close(self) -> None:
        raise NotImplementedError


def test_pinned_runtime_compatibility_defines_only_missing_debug_symbol() -> None:
    module = ModuleType("fake_sawyer_bench")

    repaired = _install_pinned_runtime_compatibility(module)

    assert repaired == ("debug_grasp_reward_pad",)
    assert module.debug_grasp_reward_pad == 0.0
    assert _install_pinned_runtime_compatibility(module) == ()


def test_close_tolerates_pinned_upstream_not_implemented() -> None:
    env = ContinualBenchStreamEnv(FakeUnclosableContinualBench())

    env.close()
