import json
from pathlib import Path

from rl_bgd.analysis.artifacts import (
    BootstrapConfig,
    bootstrap_mean_ci,
    build_paper_artifacts,
)


def _fake_run(
    root: Path,
    *,
    seed: int,
    value: float,
) -> None:
    run = root / "smoke" / f"demo__seed_{seed}"
    run.mkdir(parents=True)
    metadata = {
        "run_id": f"demo__seed_{seed}",
        "job_id": "demo",
        "suite": "smoke",
        "seed": seed,
        "git_commit": "abc123",
        "algorithm": "demo",
        "environment": "synthetic",
        "protocol": "stationary",
        "hypothesis_id": "A",
        "config_path": None,
        "primary_metric": "post_return",
        "duration_seconds": 1.5 + seed,
        "status": "success",
    }
    (run / "run_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (run / "stdout.json").write_text(
        json.dumps(
            {
                "post_return": value,
                "training": {
                    "final_10_mean_return": value - 0.5,
                },
            }
        ),
        encoding="utf-8",
    )


def test_bootstrap_mean_ci_is_deterministic() -> None:
    config = BootstrapConfig(
        samples=500,
        confidence=0.9,
        seed=9,
    )
    left = bootstrap_mean_ci(
        [1.0, 2.0, 3.0],
        config=config,
    )
    right = bootstrap_mean_ci(
        [1.0, 2.0, 3.0],
        config=config,
    )
    assert left == right
    assert left["ci_low"] <= left["mean"] <= left["ci_high"]


def test_paper_artifact_pipeline_requires_no_manual_transcription(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    for seed, value in enumerate((1.0, 2.0, 4.0)):
        _fake_run(
            run_root,
            seed=seed,
            value=value,
        )
    suite_dir = run_root / "smoke"
    (suite_dir / "suite_manifest.json").write_text(
        json.dumps(
            {
                "suite": "smoke",
                "suite_revision": 1,
                "git_commit": "abc123",
                "jobs": [1, 2, 3],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "paper"
    report = build_paper_artifacts(
        run_root,
        output,
        bootstrap=BootstrapConfig(
            samples=500,
            seed=10,
        ),
    )
    assert report["runs_aggregated"] == 3
    assert report["bootstrap_groups"] == 1
    assert report["manifests_indexed"] == 1
    for name in report["artifacts"]:
        path = output / name
        assert path.exists()
        assert path.stat().st_size > 0
