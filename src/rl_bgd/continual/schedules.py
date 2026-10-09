"""Deterministic context trajectories for nonstationary RL streams."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from numbers import Real
from typing import Literal

Context = dict[str, float]
ScheduleMode = Literal["abrupt", "smooth", "periodic", "random_walk", "recurring"]


@dataclass(frozen=True)
class ContextScheduleConfig:
    """Configuration for a context trajectory.

    All anchors must share identical keys. phase_steps controls abrupt,
    recurring, and smooth schedules. Periodic schedules use the first anchor
    as the center and the second as the amplitude. Random-walk schedules start
    from the first anchor and optionally clamp values to bounds.
    """

    mode: ScheduleMode
    anchors: tuple[Context, ...]
    phase_steps: int = 10_000
    period_steps: int = 10_000
    random_walk_std: float = 0.01
    seed: int = 0
    bounds: dict[str, tuple[float, float]] | None = None

    def validate(self) -> None:
        if self.mode not in {
            "abrupt",
            "smooth",
            "periodic",
            "random_walk",
            "recurring",
        }:
            raise ValueError(f"unsupported context schedule mode: {self.mode}")
        if not self.anchors:
            raise ValueError("at least one context anchor is required")
        if any(not isinstance(anchor, dict) for anchor in self.anchors):
            raise TypeError("context anchors must be dictionaries")
        keys = set(self.anchors[0])
        if not keys:
            raise ValueError("context anchors cannot be empty")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("context anchor keys must be non-empty strings")
        if any(set(anchor) != keys for anchor in self.anchors):
            raise ValueError("all context anchors must share identical keys")
        for anchor in self.anchors:
            for value in anchor.values():
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError("context anchor values must be real numbers")
                if not math.isfinite(float(value)):
                    raise ValueError("context anchors must contain finite values")
        if (
            isinstance(self.phase_steps, bool)
            or not isinstance(self.phase_steps, int)
            or self.phase_steps < 1
        ):
            raise ValueError("phase_steps must be a positive integer")
        if (
            isinstance(self.period_steps, bool)
            or not isinstance(self.period_steps, int)
            or self.period_steps < 1
        ):
            raise ValueError("period_steps must be a positive integer")
        if isinstance(self.random_walk_std, bool) or not isinstance(self.random_walk_std, Real):
            raise TypeError("random_walk_std must be a real number")
        if not math.isfinite(float(self.random_walk_std)) or self.random_walk_std < 0:
            raise ValueError("random_walk_std must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.mode == "smooth" and len(self.anchors) < 2:
            raise ValueError("smooth schedules require at least two anchors")
        if self.mode == "periodic" and len(self.anchors) != 2:
            raise ValueError("periodic schedules require center and amplitude anchors")
        if self.mode == "random_walk" and len(self.anchors) != 1:
            raise ValueError("random-walk schedules require exactly one starting anchor")
        if self.bounds is not None:
            if self.mode != "random_walk":
                raise ValueError("context bounds are only supported for random-walk schedules")
            if not isinstance(self.bounds, dict):
                raise TypeError("context bounds must be a dictionary")
            if set(self.bounds) != keys:
                raise ValueError("bounds keys must match context keys")
            for bounds in self.bounds.values():
                if not isinstance(bounds, tuple) or len(bounds) != 2:
                    raise TypeError("context bounds must be two-value tuples")
                low, high = bounds
                if (
                    isinstance(low, bool)
                    or isinstance(high, bool)
                    or not isinstance(low, Real)
                    or not isinstance(high, Real)
                ):
                    raise TypeError("context bounds must contain real numbers")
                if not math.isfinite(float(low)) or not math.isfinite(float(high)):
                    raise ValueError("context bounds must be finite")
                if low > high:
                    raise ValueError("context lower bound exceeds upper bound")
            start = self.anchors[0]
            for key, (low, high) in self.bounds.items():
                if not low <= start[key] <= high:
                    raise ValueError("random-walk starting anchor must lie within bounds")


class ContextSchedule:
    """Pure, task-label-free mapping from environment step to context values."""

    def __init__(self, config: ContextScheduleConfig) -> None:
        config.validate()
        self.config = config
        self._random_walk_cache: list[Context] = [dict(config.anchors[0])]

    def context_at(self, step: int) -> Context:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        mode = self.config.mode
        if mode == "abrupt":
            return self._abrupt(step)
        if mode == "recurring":
            return self._recurring(step)
        if mode == "smooth":
            return self._smooth(step)
        if mode == "periodic":
            return self._periodic(step)
        return self._random_walk(step)

    def _abrupt(self, step: int) -> Context:
        index = min(
            step // self.config.phase_steps,
            len(self.config.anchors) - 1,
        )
        return dict(self.config.anchors[index])

    def _recurring(self, step: int) -> Context:
        index = (step // self.config.phase_steps) % len(self.config.anchors)
        return dict(self.config.anchors[index])

    def _smooth(self, step: int) -> Context:
        terminal_step = self.config.phase_steps * (len(self.config.anchors) - 1)
        if step >= terminal_step:
            return dict(self.config.anchors[-1])
        segment = step // self.config.phase_steps
        within = step % self.config.phase_steps
        alpha = within / self.config.phase_steps
        left = self.config.anchors[segment]
        right = self.config.anchors[segment + 1]
        return {key: (1.0 - alpha) * left[key] + alpha * right[key] for key in left}

    def _periodic(self, step: int) -> Context:
        center, amplitude = self.config.anchors
        phase = 2.0 * math.pi * step / self.config.period_steps
        oscillation = math.sin(phase)
        return {key: center[key] + amplitude[key] * oscillation for key in center}

    def _random_walk(self, step: int) -> Context:
        while len(self._random_walk_cache) <= step:
            index = len(self._random_walk_cache)
            previous = self._random_walk_cache[-1]
            rng = random.Random(self.config.seed + index)
            current: Context = {}
            for key, value in previous.items():
                updated = value + rng.gauss(
                    0.0,
                    self.config.random_walk_std,
                )
                if self.config.bounds is not None:
                    low, high = self.config.bounds[key]
                    updated = min(max(updated, low), high)
                current[key] = updated
            self._random_walk_cache.append(current)
        return dict(self._random_walk_cache[step])
