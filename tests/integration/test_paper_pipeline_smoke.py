from pathlib import Path

import pytest

from rl_bgd.analysis.paper_artifacts import (
    PaperArtifactConfig,
    build_paper_artifacts,
)
from rl_bgd.artifacts import (
    discover_run_directories,
    load_run_directory,
)
from rl_bgd.experiments.suites import execute_suite


@pytest.mark.slow
def test_smoke_suite_to_paper_artifacts_end_to_end(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    execution = execute_suite(
        "smoke",
        results_root,
    )
    assert execution["status"] == "success"
    assert execution["failures"] == []

    run_dirs = discover_run_directories(results_root)
    assert len(run_dirs) == 3
    loaded = [load_run_directory(path) for path in run_dirs]
    assert {run.manifest.method for run in loaded} == {
        "SAC-Adam",
        "SAC-BGD",
        "PPO-Adam",
    }

    paper_dir = tmp_path / "paper"
    manifest = build_paper_artifacts(
        results_root,
        paper_dir,
        config=PaperArtifactConfig(
            confidence=0.9,
            bootstrap_resamples=200,
            seed=170,
            figure_formats=("png",),
        ),
    )
    assert manifest["run_count"] == 3
    assert manifest["integrity"]["manual_result_transcription"] is False
    assert manifest["integrity"]["incomplete_runs_allowed"] is False
    assert (paper_dir / "paper_manifest.json").is_file()
    expected_table_stems = {
        "aggregate_statistics",
        "hierarchical_task_statistics",
        "paired_differences",
        "results_summary",
        "information_access",
        "run_index",
    }
    generated_tables = set(manifest["generated_tables"])
    for stem in expected_table_stems:
        for extension in (
            "csv",
            "md",
            "tex",
        ):
            relative = f"tables/{stem}.{extension}"
            assert relative in generated_tables
            assert (paper_dir / relative).is_file()

    source_runs = manifest["source_runs"]
    assert len(source_runs) == 3
    for source in source_runs:
        hashes = source["source_hashes"]
        assert set(hashes) == {
            "manifest.json",
            "config.yaml",
            "metrics.csv",
            "summary.json",
        }
        assert all(len(value) == 64 for value in hashes.values())

    generated_figures = set(manifest["generated_figures"])
    assert "figures/final_performance.png" in generated_figures
    assert "figures/compute_performance.png" in generated_figures
    assert (paper_dir / "figures" / "final_performance.png").is_file()
    assert (paper_dir / "figures" / "compute_performance.png").is_file()
