import torch

from rl_bgd.agents.ppo.recurrent_rollout import (
    RecurrentRolloutBuffer,
)


def test_recurrent_rollout_preserves_chunk_initial_hidden_and_resets() -> None:
    buffer = RecurrentRolloutBuffer(
        4,
        1,
        1,
        2,
        2,
    )
    for index in range(4):
        buffer.add(
            torch.tensor([float(index)]),
            torch.tensor([0.0]),
            1.0,
            terminated=False,
            truncated=False,
            value=torch.tensor([0.0]),
            next_value=torch.tensor([0.0]),
            log_prob=torch.tensor([0.0]),
            episode_start=(
                index
                in {
                    0,
                    2,
                }
            ),
            actor_hidden=torch.tensor(
                [
                    float(index),
                    1.0,
                ]
            ),
            value_hidden=torch.tensor(
                [
                    float(index),
                    2.0,
                ]
            ),
        )
    buffer.compute_gae(
        gamma=0.99,
        gae_lambda=0.95,
    )
    generator = torch.Generator().manual_seed(0)
    batches = list(
        buffer.sequence_batches(
            2,
            generator=generator,
        )
    )
    assert len(batches) == 2
    starts = sorted(
        int(
            batch.observations[
                0,
                0,
            ].item()
        )
        for batch in batches
    )
    assert starts == [
        0,
        2,
    ]
    for batch in batches:
        start = int(
            batch.observations[
                0,
                0,
            ].item()
        )
        assert batch.initial_actor_hidden[0].item() == float(start)
        assert bool(
            batch.episode_starts[
                0,
                0,
            ]
        )
