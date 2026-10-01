import torch

from rl_bgd.models.recurrent_sac import (
    RecurrentQNetwork,
    RecurrentSACActor,
)


def test_recurrent_sac_actor_reset_breaks_prior_history() -> None:
    torch.manual_seed(80)
    actor = RecurrentSACActor(
        1,
        1,
        action_low=torch.tensor(
            [-1.0]
        ),
        action_high=torch.tensor(
            [1.0]
        ),
        recurrent_hidden_dim=8,
        encoder_hidden_dims=(8,),
    )
    starts = torch.tensor(
        [
            [
                [True],
                [False],
                [True],
                [False],
            ]
        ]
    )
    left = torch.tensor(
        [
            [
                [1.0],
                [2.0],
                [3.0],
                [4.0],
            ]
        ]
    )
    right = torch.tensor(
        [
            [
                [-9.0],
                [20.0],
                [3.0],
                [4.0],
            ]
        ]
    )
    left_hidden, _ = (
        actor.hidden_sequence(
            left,
            starts,
        )
    )
    right_hidden, _ = (
        actor.hidden_sequence(
            right,
            starts,
        )
    )
    torch.testing.assert_close(
        left_hidden[
            :,
            2:,
        ],
        right_hidden[
            :,
            2:,
        ],
    )


def test_recurrent_q_sequence_and_next_state_shapes() -> None:
    torch.manual_seed(81)
    critic = RecurrentQNetwork(
        2,
        1,
        recurrent_hidden_dim=6,
        encoder_hidden_dims=(6,),
        q_hidden_dims=(8,),
    )
    observations = torch.randn(
        3,
        5,
        2,
    )
    actions = torch.randn(
        3,
        5,
        1,
    )
    starts = torch.zeros(
        3,
        5,
        1,
        dtype=torch.bool,
    )
    starts[
        :,
        0,
    ] = True
    hidden, _ = (
        critic.hidden_sequence(
            observations,
            starts,
        )
    )
    q = critic.q_from_hidden(
        hidden,
        actions,
    )
    next_q, next_hidden = (
        critic.next_q_from_hidden(
            observations + 0.1,
            actions,
            hidden,
        )
    )
    assert q.shape == (
        3,
        5,
        1,
    )
    assert next_q.shape == q.shape
    assert next_hidden.shape == (
        3,
        5,
        6,
    )
