"""Strict hidden-task adapter for sail-sg/ContinualBench."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from shutil import copy2
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from rl_bgd.envs.synthetic.lqr import TensorBox

CONTINUAL_BENCH_TASKS: tuple[str, ...] = (
    "button",
    "door",
    "window",
    "faucet",
    "peg",
    "block",
)
SwitchMode = Literal[
    "success",
    "fixed_steps",
    "success_or_budget",
]


class ContinualBenchImportError(ImportError):
    """Raised when the optional ContinualBench dependency is unavailable."""


def _install_pinned_runtime_compatibility(
    sawyer_bench_module: Any,
) -> tuple[str, ...]:
    """Patch non-semantic defects in the pinned ContinualBench revision.

    The pinned upstream block reward returns a debug-only symbol whose
    calculation is commented out. Defining that missing module global lets the
    original reward function finish without replacing any reward computation.
    """

    repaired: list[str] = []
    if not hasattr(
        sawyer_bench_module,
        "debug_grasp_reward_pad",
    ):
        sawyer_bench_module.debug_grasp_reward_pad = 0.0
        repaired.append(
            "debug_grasp_reward_pad"
        )
    return tuple(
        repaired
    )


_ASSET_FILE_PATTERN = re.compile(r"""\bfile\s*=\s*["']([^"']+)["']""")
_XML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def _repair_missing_metaworld_assets(
    continual_bench_envs: Any,
    metaworld_module: Any,
) -> tuple[str, ...]:
    """Restore missing XML-referenced assets from canonical Meta-World.

    Existing ContinualBench assets are never overwritten. Each missing file is
    copied from the identical relative path under Meta-World's assets tree.
    When the installed benchmark root is present, only its reachable XML graph
    is traversed. Copied XML files are scanned too, so unrelated stale package
    XML cannot block the active benchmark.
    """

    continual_file = getattr(continual_bench_envs, "__file__", None)
    metaworld_file = getattr(metaworld_module, "__file__", None)
    if continual_file is None or metaworld_file is None:
        raise ContinualBenchImportError(
            "cannot locate installed benchmark packages for asset repair"
        )

    destination_root = Path(continual_file).resolve().parent / "assets"
    source_root = Path(metaworld_file).resolve().parent / "assets"
    if not destination_root.is_dir():
        raise ContinualBenchImportError(
            "installed ContinualBench package does not contain an assets directory"
        )
    if not source_root.is_dir():
        raise ContinualBenchImportError(
            "installed Meta-World package does not contain an assets directory"
        )

    destination_resolved = destination_root.resolve()
    benchmark_root = (
        destination_root
        / "sawyer_xyz"
        / "sawyer_bench.xml"
    )
    queue = (
        [benchmark_root]
        if benchmark_root.is_file()
        else sorted(destination_root.rglob("*.xml"))
    )
    visited: set[Path] = set()
    repaired: list[str] = []

    while queue:
        xml_path = queue.pop(0).resolve()
        if xml_path in visited:
            continue
        visited.add(xml_path)
        try:
            contents = xml_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise ContinualBenchImportError(
                f"cannot read ContinualBench asset XML: {xml_path}"
            ) from exc

        live_contents = _XML_COMMENT_PATTERN.sub(
            "",
            contents,
        )
        for raw_reference in _ASSET_FILE_PATTERN.findall(live_contents):
            target = (xml_path.parent / raw_reference).resolve()
            try:
                relative = target.relative_to(destination_resolved)
            except ValueError:
                continue

            if target.exists():
                if target.suffix.lower() == ".xml":
                    queue.append(target)
                continue

            canonical = source_root / relative
            if (
                not canonical.is_file()
                and len(relative.parts) >= 2
                and relative.parts[0] == relative.parts[1]
            ):
                canonical = source_root.joinpath(*relative.parts[1:])
            if not canonical.is_file() and "textures" in relative.parts:
                canonical = source_root / "textures" / relative.name
            if not canonical.is_file():
                raise ContinualBenchImportError(
                    "ContinualBench references a missing asset and no canonical "
                    "Meta-World source is available for "
                    f"{relative.as_posix()}"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(canonical, target)
            repaired.append(relative.as_posix())
            if target.suffix.lower() == ".xml":
                queue.append(target)

    return tuple(repaired)


@dataclass(frozen=True)
class ContinualBenchStreamConfig:
    """Task-stream controls for the unified ContinualBench world."""

    task_sequence: tuple[str, ...] = CONTINUAL_BENCH_TASKS
    switch_mode: SwitchMode = "success_or_budget"
    steps_per_task: int = 10_000
    strict_task_agnostic: bool = True

    def validate(self) -> None:
        if not self.task_sequence:
            raise ValueError("ContinualBench task sequence cannot be empty")
        if len(set(self.task_sequence)) != len(self.task_sequence):
            raise ValueError("ContinualBench task sequence must not contain duplicates")
        unknown = set(self.task_sequence) - set(CONTINUAL_BENCH_TASKS)
        if unknown:
            raise ValueError(f"unsupported ContinualBench tasks: {sorted(unknown)}")
        if self.switch_mode not in {
            "success",
            "fixed_steps",
            "success_or_budget",
        }:
            raise ValueError(f"unsupported ContinualBench switch mode: {self.switch_mode}")
        if self.steps_per_task < 1:
            raise ValueError("steps_per_task must be positive")


class ContinualBenchStreamEnv:
    """Select the active reward/task without exposing task identity to the agent.

    ContinualBench uses one unified MuJoCo world and emits a reward dictionary
    for all six benchmark tasks. This wrapper chooses the active reward
    internally. Source revisions currently use a legacy four-value step API,
    while the README documents a five-value Gymnasium API; both are accepted.

    A hidden task switch does not itself emit termination/truncation. When the
    external environment naturally ends on the same step, the switch is
    deferred until the trainer's next reset.
    """

    def __init__(
        self,
        env: Any,
        *,
        config: ContinualBenchStreamConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or ContinualBenchStreamConfig()
        self.config.validate()
        self.env = env
        self.device = torch.device(device)
        self.task_index = 0
        self.task_step = 0
        self.environment_step = 0
        self._switch_on_reset = False

        available = tuple(
            getattr(
                env,
                "all_tasks",
                CONTINUAL_BENCH_TASKS,
            )
        )
        missing = set(self.config.task_sequence) - set(available)
        if missing:
            raise ValueError(
                f"external ContinualBench environment is missing tasks: {sorted(missing)}"
            )
        self.action_space = self._tensor_box(env.action_space)
        self.observation_space = self._tensor_box(env.observation_space)
        self._set_active_task()

    @property
    def active_task(self) -> str:
        return self.config.task_sequence[self.task_index]

    @property
    def evaluation_context(self) -> dict[str, int | str]:
        """Evaluator-only stream metadata; never returned in strict mode."""

        return {
            "task_index": self.task_index,
            "task_name": self.active_task,
            "task_step": self.task_step,
            "environment_step": self.environment_step,
        }

    def _tensor_box(self, space: Any) -> TensorBox:
        if not hasattr(space, "low") or not hasattr(space, "high"):
            raise TypeError("ContinualBench adapter requires continuous Box spaces")
        return TensorBox(
            low=torch.as_tensor(
                np.asarray(space.low),
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1),
            high=torch.as_tensor(
                np.asarray(space.high),
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1),
        )

    def _set_active_task(self) -> None:
        setter = getattr(self.env, "set_task", None)
        if setter is None:
            raise TypeError("ContinualBench environment does not expose set_task")
        setter(self.active_task)

    def _external_reset(
        self,
        *,
        seed: int | None = None,
    ) -> Tensor:
        try:
            output = self.env.reset(seed=seed)
        except TypeError:
            output = self.env.reset()
        observation = output[0] if isinstance(output, tuple) else output
        return self._observation(observation)

    def _observation(self, value: Any) -> Tensor:
        tensor = torch.as_tensor(
            np.asarray(value),
            dtype=torch.float32,
            device=self.device,
        )
        return tensor.reshape(-1)

    def _reward(self, payload: Any) -> float:
        if isinstance(payload, Mapping):
            if self.active_task not in payload:
                raise KeyError(f"reward mapping lacks active task {self.active_task!r}")
            payload = payload[self.active_task]
        value = float(payload)
        if not np.isfinite(value):
            raise FloatingPointError("ContinualBench returned a nonfinite reward")
        return value

    def _success(self, info: Any) -> bool:
        if not isinstance(info, Mapping):
            return False
        task_info = info.get(self.active_task)
        if isinstance(task_info, Mapping):
            return bool(task_info.get("success", False))
        return bool(info.get("success", False))

    def _should_switch(self, *, success: bool) -> bool:
        mode = self.config.switch_mode
        budget = self.task_step >= self.config.steps_per_task
        if mode == "success":
            return success
        if mode == "fixed_steps":
            return budget
        return success or budget

    def _has_next_task(self) -> bool:
        return self.task_index + 1 < len(self.config.task_sequence)

    def _advance_task(self) -> None:
        if not self._has_next_task():
            return
        self.task_index += 1
        self.task_step = 0
        self._switch_on_reset = False
        self._set_active_task()

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        if self._switch_on_reset:
            self._advance_task()
        observation = self._external_reset(seed=seed)
        if self.config.strict_task_agnostic:
            return observation, {}
        return observation, {
            "task_name": self.active_task,
            "task_index": self.task_index,
        }

    def step(
        self,
        action: Tensor,
    ) -> tuple[Tensor, float, bool, bool, dict[str, object]]:
        numpy_action = (
            action.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .numpy()
            .reshape(self.env.action_space.shape)
        )
        output = self.env.step(numpy_action)
        if not isinstance(output, tuple):
            raise RuntimeError("ContinualBench step output must be a tuple")
        if len(output) == 4:
            observation, reward_payload, truncated, info = output
            terminated = False
        elif len(output) == 5:
            (
                observation,
                reward_payload,
                terminated,
                truncated,
                info,
            ) = output
        else:
            raise RuntimeError(f"unexpected ContinualBench step signature length: {len(output)}")

        reward = self._reward(reward_payload)
        success = self._success(info)
        self.environment_step += 1
        self.task_step += 1
        should_switch = self._should_switch(success=success) and self._has_next_task()

        next_observation = self._observation(observation)
        if should_switch:
            if bool(terminated) or bool(truncated):
                self._switch_on_reset = True
            else:
                self._advance_task()
                next_observation = self._external_reset()

        if self.config.strict_task_agnostic:
            safe_info: dict[str, object] = {}
        else:
            safe_info = {
                "task_name": self.active_task,
                "task_index": self.task_index,
                "task_success": success,
            }
        return (
            next_observation,
            reward,
            bool(terminated),
            bool(truncated),
            safe_info,
        )

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is None:
            return
        try:
            close()
        except NotImplementedError:
            # The pinned ContinualBench base environment defines close() only
            # as a NotImplementedError stub. With no renderer allocated there
            # is no benchmark-owned resource for this wrapper to release.
            return


def make_continual_bench_stream(
    *,
    config: ContinualBenchStreamConfig | None = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
    render_mode: str | None = None,
) -> ContinualBenchStreamEnv:
    """Create the pinned ContinualBench environment lazily."""

    try:
        envs = import_module("continual_bench.envs")
        sawyer_bench = import_module(
            "continual_bench.envs.mujoco.sawyer_bench"
        )
        metaworld = import_module("metaworld")
    except ImportError as exc:
        raise ContinualBenchImportError(
            "ContinualBench is optional. Install the 'continual-bench' extra."
        ) from exc
    _repair_missing_metaworld_assets(
        envs,
        metaworld,
    )
    _install_pinned_runtime_compatibility(
        sawyer_bench
    )
    env_class = getattr(envs, "ContinualBenchEnv", None)
    if env_class is None:
        raise ContinualBenchImportError("installed package does not expose ContinualBenchEnv")
    env = env_class(
        seed=seed,
        render_mode=render_mode,
    )
    return ContinualBenchStreamEnv(
        env,
        config=config,
        device=device,
    )
