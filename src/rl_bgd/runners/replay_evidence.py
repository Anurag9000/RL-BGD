"""Synthetic replay-reuse experiment for posterior uncertainty."""

from __future__ import annotations

import json

import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDLoss, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior
from rl_bgd.replay.buffer import ReplayBatch
from rl_bgd.replay.evidence_accounting import (
    ReplayEvidenceConfig,
    replay_evidence_weights,
    weighted_evidence_mean,
)


class Scalar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(
            torch.tensor(
                [0.8],
                dtype=torch.float32,
            )
        )


def _metadata_batch(
    usage_count: int,
    fresh: bool,
) -> ReplayBatch:
    one = torch.ones(1, 1)
    zero_bool = torch.zeros(
        1,
        1,
        dtype=torch.bool,
    )
    return ReplayBatch(
        observations=one,
        actions=one,
        rewards=one,
        next_observations=one,
        terminated=zero_bool,
        truncated=zero_bool,
        transition_ids=torch.zeros(
            1,
            1,
            dtype=torch.long,
        ),
        insertion_steps=torch.zeros(
            1,
            1,
            dtype=torch.long,
        ),
        usage_counts=torch.full(
            (1, 1),
            usage_count,
            dtype=torch.long,
        ),
        fresh=torch.full(
            (1, 1),
            fresh,
            dtype=torch.bool,
        ),
    )


def run_replay_evidence_demo(
    *,
    uses: int = 30,
    seed: int = 0,
) -> dict[str, float]:
    if uses < 1:
        raise ValueError(
            "uses must be >= 1"
        )
    outputs: dict[str, float] = {}
    for mode in (
        "all_replay",
        "fresh_only_uncertainty",
        "inverse_reuse_weight",
        "normalized_batch_evidence",
    ):
        torch.manual_seed(seed)
        module = Scalar()
        posterior = (
            DiagonalGaussianPosterior.from_module(
                module,
                prior_std=0.3,
            )
        )
        updater = BGDUpdater(
            posterior,
            BGDConfig(
                eta=0.1,
                mc_samples=8,
                antithetic=True,
            ),
        )
        for use in range(
            1,
            uses + 1,
        ):
            batch = _metadata_batch(
                use,
                fresh=(use == 1),
            )
            evidence = replay_evidence_weights(
                batch,
                ReplayEvidenceConfig(  # type: ignore[arg-type]
                    mode=mode
                ),
            )
            evidence_weights = evidence.weights

            def objective(
                params: dict[str, torch.Tensor],
            ) -> BGDLoss:
                per_item = (
                    0.5
                    * params["w"]
                    .square()
                    .reshape(1, 1)
                )
                return BGDLoss(
                    mean=per_item.mean(),
                    uncertainty=weighted_evidence_mean(
                        per_item,
                        evidence_weights,
                    ),
                )

            updater.step(objective)
        outputs[
            f"{mode}_sigma"
        ] = float(
            posterior.stds["w"].item()
        )
        outputs[
            f"{mode}_abs_mean"
        ] = float(
            posterior.means["w"]
            .abs()
            .item()
        )
    return outputs


def main() -> None:
    print(
        json.dumps(
            run_replay_evidence_demo(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
