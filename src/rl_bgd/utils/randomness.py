"""Central randomness controls for reproducible experiments."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
import torch

from rl_bgd.utils.checkpoint_progress import checkpoint_integer


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


def _validated_torch_rng_state(
    value: Any,
    *,
    name: str,
    device: torch.device | str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} RNG checkpoint must be a tensor")
    candidate = value.detach().cpu().contiguous()
    if candidate.dtype != torch.uint8 or candidate.ndim != 1:
        raise ValueError(f"{name} RNG checkpoint must be a one-dimensional uint8 tensor")
    generator = torch.Generator(device=device)
    try:
        generator.set_state(candidate)
    except RuntimeError as exc:
        raise ValueError(f"invalid {name} RNG checkpoint state") from exc
    return candidate


def load_random_state_dict(state: dict[str, Any]) -> None:
    """Restore validated process RNG streams captured by :func:`random_state_dict`."""

    if checkpoint_integer(state.get("version"), name="random-state version") != 1:
        raise ValueError("unsupported random-state checkpoint version")

    python_state = state.get("python")
    if not isinstance(python_state, tuple):
        raise ValueError("invalid Python RNG checkpoint state")
    python_validator = random.Random()
    try:
        python_validator.setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Python RNG checkpoint state") from exc

    numpy_state = state.get("numpy")
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise ValueError("invalid NumPy RNG checkpoint state")
    typed_numpy_state = cast(tuple[str, np.ndarray, int, int, float], numpy_state)
    numpy_validator = np.random.RandomState()
    try:
        numpy_validator.set_state(typed_numpy_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid NumPy RNG checkpoint state") from exc

    torch_cpu = _validated_torch_rng_state(
        state.get("torch_cpu"),
        name="torch CPU",
        device="cpu",
    )

    cuda_state = state.get("torch_cuda")
    checked_cuda: list[torch.Tensor] | None = None
    if cuda_state is not None:
        if not isinstance(cuda_state, list) or not all(
            isinstance(item, torch.Tensor) for item in cuda_state
        ):
            raise TypeError("CUDA RNG checkpoint must be a list of tensors")
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError("CUDA RNG checkpoint device count mismatch")
        checked_cuda = [
            _validated_torch_rng_state(
                item,
                name=f"CUDA device {index}",
                device=torch.device("cuda", index),
            )
            for index, item in enumerate(cuda_state)
        ]

    random.setstate(python_state)
    np.random.set_state(typed_numpy_state)
    torch.set_rng_state(torch_cpu)
    if checked_cuda is not None:
        torch.cuda.set_rng_state_all(checked_cuda)
