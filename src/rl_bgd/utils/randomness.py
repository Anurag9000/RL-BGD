"""Central randomness controls for reproducible experiments."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


@contextmanager
def preserved_random_state() -> Iterator[None]:
    """Run evaluator-only work without perturbing global training RNG streams."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def random_state_dict() -> dict[str, Any]:
    """Capture process RNG streams needed for deterministic local resume."""

    return {
        "version": 1,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def load_random_state_dict(state: dict[str, Any]) -> None:
    """Restore process RNG streams captured by :func:`random_state_dict`."""

    if state.get("version") != 1:
        raise ValueError("unsupported random-state checkpoint version")
    torch_cpu = state.get("torch_cpu")
    if not isinstance(torch_cpu, torch.Tensor):
        raise TypeError("torch CPU RNG checkpoint must be a tensor")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch_cpu.cpu())

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if not isinstance(cuda_state, list) or not all(
            isinstance(item, torch.Tensor) for item in cuda_state
        ):
            raise TypeError("CUDA RNG checkpoint must be a list of tensors")
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError("CUDA RNG checkpoint device count mismatch")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])
