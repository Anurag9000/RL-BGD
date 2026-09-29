"""Analytical continual quadratic optimization benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

StreamMode = Literal["abrupt", "smooth", "recurring"]


@dataclass(frozen=True)
class QuadraticTask:
    """Positive-definite quadratic objective.

    L(theta) = 0.5 * (theta - optimum)^T H (theta - optimum)
    """

    optimum: Tensor
    hessian: Tensor
    name: str = "quadratic"

    def __post_init__(self) -> None:
        if self.optimum.ndim != 1:
            raise ValueError("optimum must be a vector")
        d = self.optimum.numel()
        if self.hessian.shape != (d, d):
            raise ValueError("hessian must have shape (d, d)")
        if not torch.allclose(
            self.hessian, self.hessian.T, atol=1e-6, rtol=1e-6
        ):
            raise ValueError("hessian must be symmetric")
        eigvals = torch.linalg.eigvalsh(self.hessian.float())
        if torch.any(eigvals <= 0):
            raise ValueError("hessian must be positive definite")

    @property
    def dimension(self) -> int:
        return self.optimum.numel()

    def loss(self, theta: Tensor) -> Tensor:
        if theta.shape != self.optimum.shape:
            raise ValueError("theta shape does not match task dimension")
        delta = theta - self.optimum.to(
            device=theta.device, dtype=theta.dtype
        )
        hessian = self.hessian.to(
            device=theta.device, dtype=theta.dtype
        )
        return 0.5 * delta @ hessian @ delta

    def gradient(self, theta: Tensor) -> Tensor:
        if theta.shape != self.optimum.shape:
            raise ValueError("theta shape does not match task dimension")
        hessian = self.hessian.to(
            device=theta.device, dtype=theta.dtype
        )
        return hessian @ (
            theta - self.optimum.to(
                device=theta.device, dtype=theta.dtype
            )
        )


def diagonal_quadratic(
    dimension: int,
    *,
    optimum: float | Tensor = 0.0,
    curvature: float | Tensor = 1.0,
    name: str = "diagonal",
    device: torch.device | str = "cpu",
) -> QuadraticTask:
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    if isinstance(optimum, Tensor):
        opt = optimum.detach().float().to(device)
        if opt.shape != (dimension,):
            raise ValueError("optimum tensor has wrong shape")
    else:
        opt = torch.full((dimension,), float(optimum), device=device)
    if isinstance(curvature, Tensor):
        diag = curvature.detach().float().to(device)
        if diag.shape != (dimension,):
            raise ValueError("curvature tensor has wrong shape")
    else:
        diag = torch.full(
            (dimension,), float(curvature), device=device
        )
    if torch.any(diag <= 0):
        raise ValueError("curvature must be positive")
    return QuadraticTask(
        optimum=opt, hessian=torch.diag(diag), name=name
    )


def rotated_quadratic(
    dimension: int,
    *,
    eigenvalues: Tensor | None = None,
    optimum: Tensor | None = None,
    seed: int = 0,
    name: str = "rotated",
    device: torch.device | str = "cpu",
) -> QuadraticTask:
    if dimension < 2:
        raise ValueError("rotated quadratic requires dimension >= 2")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(
        (dimension, dimension),
        generator=generator,
        dtype=torch.float32,
    )
    q, _ = torch.linalg.qr(matrix)
    if eigenvalues is None:
        eig = torch.linspace(0.5, 2.0, dimension)
    else:
        eig = eigenvalues.detach().float().cpu()
        if eig.shape != (dimension,) or torch.any(eig <= 0):
            raise ValueError(
                "eigenvalues must be a positive vector of length dimension"
            )
    hessian = (q @ torch.diag(eig) @ q.T).to(device)
    if optimum is None:
        opt = torch.zeros(dimension, device=device)
    else:
        opt = optimum.detach().float().to(device)
        if opt.shape != (dimension,):
            raise ValueError("optimum tensor has wrong shape")
    return QuadraticTask(
        optimum=opt, hessian=hessian, name=name
    )


class QuadraticStream:
    """Nonstationary stream of quadratic objectives.

    Ground-truth segment identity is evaluation metadata. Optimization code
    consumes only task_at(step) and needs no boundary callback.
    """

    def __init__(
        self,
        tasks: list[QuadraticTask],
        *,
        segment_steps: int,
        mode: StreamMode = "abrupt",
    ) -> None:
        if not tasks:
            raise ValueError("at least one task is required")
        if segment_steps < 1:
            raise ValueError("segment_steps must be >= 1")
        if len({task.dimension for task in tasks}) != 1:
            raise ValueError("all tasks must share dimension")
        if mode not in {"abrupt", "smooth", "recurring"}:
            raise ValueError(f"unsupported stream mode: {mode}")
        self.tasks = tasks
        self.segment_steps = segment_steps
        self.mode = mode

    @property
    def dimension(self) -> int:
        return self.tasks[0].dimension

    def segment_index(self, step: int) -> int:
        if step < 0:
            raise ValueError("step must be non-negative")
        raw = step // self.segment_steps
        if self.mode == "recurring":
            return raw % len(self.tasks)
        return min(raw, len(self.tasks) - 1)

    def task_at(self, step: int) -> QuadraticTask:
        index = self.segment_index(step)
        if self.mode != "smooth" or index >= len(self.tasks) - 1:
            return self.tasks[index]
        local = step % self.segment_steps
        alpha = local / self.segment_steps
        left = self.tasks[index]
        right = self.tasks[index + 1]
        optimum = (
            (1.0 - alpha) * left.optimum + alpha * right.optimum
        )
        hessian = (
            (1.0 - alpha) * left.hessian + alpha * right.hessian
        )
        return QuadraticTask(
            optimum=optimum,
            hessian=hessian,
            name=f"{left.name}->{right.name}@{alpha:.3f}",
        )

    def is_evaluation_boundary(self, step: int) -> bool:
        return step > 0 and step % self.segment_steps == 0
