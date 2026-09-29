"""Analytical synthetic benchmarks."""

from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv, TensorBox
from rl_bgd.envs.synthetic.nonstationary_lqr import ScheduledLQREnv
from rl_bgd.envs.synthetic.quadratic import (
    QuadraticStream,
    QuadraticTask,
    diagonal_quadratic,
    rotated_quadratic,
)

__all__ = [
    "LinearQuadraticControlEnv",
    "QuadraticStream",
    "QuadraticTask",
    "ScheduledLQREnv",
    "TensorBox",
    "diagonal_quadratic",
    "rotated_quadratic",
]
