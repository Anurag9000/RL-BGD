from pathlib import Path

from rl_bgd.runners.sac_lqr import run_sac_lqr


def test_stationary_sac_runner_exposes_resume_controls(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stationary_sac.pt"
    common = {
        "steps": 48,
        "seed": 123,
        "device": "cpu",
        "horizon": 8,
        "hidden_dims": (8, 8),
        "random_steps": 12,
        "batch_size": 8,
        "replay_capacity": 64,
        "evaluation_episodes": 1,
        "evaluation_seed": 2_000,
    }
    partial = run_sac_lqr(
        **common,
        checkpoint_path=str(checkpoint),
        checkpoint_interval=12,
        max_steps_this_call=24,
    )
    assert partial["training"]["completed"] is False
    assert partial["training"]["steps"] == 24
    assert checkpoint.is_file()

    resumed = run_sac_lqr(
        **common,
        resume_from=str(checkpoint),
    )
    assert resumed["training"]["completed"] is True
    assert resumed["training"]["steps"] == 48
