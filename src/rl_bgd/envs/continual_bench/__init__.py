"""Adapters for sail-sg/ContinualBench."""

from rl_bgd.envs.continual_bench.stream import (
    CONTINUAL_BENCH_TASKS,
    ContinualBenchImportError,
    ContinualBenchStreamConfig,
    ContinualBenchStreamEnv,
    make_continual_bench_stream,
)

__all__ = [
    "CONTINUAL_BENCH_TASKS",
    "ContinualBenchImportError",
    "ContinualBenchStreamConfig",
    "ContinualBenchStreamEnv",
    "make_continual_bench_stream",
]
