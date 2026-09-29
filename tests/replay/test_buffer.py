import torch

from rl_bgd.replay.buffer import ReplayBuffer


def test_replay_tracks_usage_freshness_and_truncation_bootstrap() -> None:
    buffer = ReplayBuffer(4, 1, 1)
    for i in range(4):
        buffer.add(
            torch.tensor([float(i)]),
            torch.tensor([0.0]),
            1.0,
            torch.tensor([float(i + 1)]),
            terminated=(i == 0),
            truncated=(i == 1),
            insertion_step=i,
        )
    generator = torch.Generator().manual_seed(0)
    batch = buffer.sample(
        4, generator=generator
    )
    assert torch.all(
        batch.usage_counts >= 1
    )
    assert torch.all(batch.fresh)
    mask = (
        batch.truncated
        & ~batch.terminated
    )
    assert torch.all(
        batch.bootstrap_mask[mask] == 1
    )
    assert torch.all(
        batch.bootstrap_mask[
            batch.terminated
        ]
        == 0
    )


def test_replay_checkpoint_round_trip() -> None:
    source = ReplayBuffer(4, 1, 1)
    source.add(
        torch.tensor([1.0]),
        torch.tensor([0.25]),
        2.0,
        torch.tensor([1.5]),
        terminated=False,
        truncated=True,
        insertion_step=7,
    )
    state = source.state_dict()
    restored = ReplayBuffer(4, 1, 1)
    restored.load_state_dict(state)
    assert len(restored) == 1
    torch.testing.assert_close(
        restored.observations[0],
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(
        restored.actions[0],
        torch.tensor([0.25]),
    )
    assert restored.truncated[0, 0]
