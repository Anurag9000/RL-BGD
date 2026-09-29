import torch

from rl_bgd.envs.synthetic.quadratic import (
    QuadraticStream,
    diagonal_quadratic,
    rotated_quadratic,
)


def test_diagonal_quadratic_exact_loss_and_gradient() -> None:
    task = diagonal_quadratic(
        2,
        optimum=torch.tensor([1.0, -1.0]),
        curvature=torch.tensor([2.0, 4.0]),
    )
    theta = torch.tensor([2.0, 1.0])
    torch.testing.assert_close(
        task.gradient(theta), torch.tensor([2.0, 8.0])
    )
    torch.testing.assert_close(
        task.loss(theta), torch.tensor(9.0)
    )


def test_rotated_quadratic_is_dense_spd() -> None:
    task = rotated_quadratic(8, seed=3)
    assert torch.all(
        torch.linalg.eigvalsh(task.hessian) > 0
    )
    off_diagonal = task.hessian - torch.diag(
        torch.diag(task.hessian)
    )
    assert torch.count_nonzero(off_diagonal) > 0


def test_abrupt_and_recurring_schedule() -> None:
    tasks = [
        diagonal_quadratic(
            2, optimum=float(i), name=str(i)
        )
        for i in range(3)
    ]
    abrupt = QuadraticStream(
        tasks, segment_steps=5, mode="abrupt"
    )
    assert abrupt.task_at(0).name == "0"
    assert abrupt.task_at(5).name == "1"
    assert abrupt.task_at(100).name == "2"
    recurring = QuadraticStream(
        tasks, segment_steps=5, mode="recurring"
    )
    assert recurring.task_at(15).name == "0"
    assert recurring.task_at(20).name == "1"


def test_smooth_schedule_interpolates() -> None:
    a = diagonal_quadratic(
        1, optimum=0.0, curvature=1.0, name="a"
    )
    b = diagonal_quadratic(
        1, optimum=2.0, curvature=3.0, name="b"
    )
    stream = QuadraticStream(
        [a, b], segment_steps=10, mode="smooth"
    )
    middle = stream.task_at(5)
    torch.testing.assert_close(
        middle.optimum, torch.tensor([1.0])
    )
    torch.testing.assert_close(
        middle.hessian, torch.tensor([[2.0]])
    )
