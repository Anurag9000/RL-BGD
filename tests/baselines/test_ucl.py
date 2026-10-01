import torch

from rl_bgd.baselines.ucl import (
    UCLRegularizationConfig,
    snapshot_ucl_layers,
    ucl_regularization,
)
from rl_bgd.models.ucl_ppo import UCLBayesianLinear


def test_ucl_penalty_is_finite_and_boundary_l1_activates() -> None:
    torch.manual_seed(141)
    layers = (
        UCLBayesianLinear(3, 4),
        UCLBayesianLinear(4, 2),
    )
    snapshots = snapshot_ucl_layers(layers)
    with torch.no_grad():
        layers[0].weight_mu.add_(0.05)
        layers[1].bias_mu.sub_(0.03)

    before = ucl_regularization(
        layers,
        snapshots,
        minibatch_size=8,
        config=UCLRegularizationConfig(),
        saved_task=False,
    )
    after = ucl_regularization(
        layers,
        snapshots,
        minibatch_size=8,
        config=UCLRegularizationConfig(),
        saved_task=True,
    )
    assert torch.isfinite(before["total"])
    assert torch.isfinite(after["total"])
    assert before["mean_l1"].item() == 0.0
    assert after["mean_l1"].item() > 0.0
    assert after["total"].item() > before["total"].item()
