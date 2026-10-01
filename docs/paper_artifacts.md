# Paper artifact pipeline

Phase 15 consumes only canonical completed run directories created by the
Phase 14 suite launcher.

Run:

    python scripts/build_paper_artifacts.py \
        --run-root artifacts/suites \
        --output-dir artifacts/paper

## Canonical raw-run contract

Every completed run is authoritative only when its directory contains:

- manifest.json;
- config.yaml;
- metrics.csv;
- summary.json.

The strict loader validates schema versions, run IDs, completion state, source
files, and provenance hashes before a run can enter paper aggregation.
stdout.json, stderr.log, and execution_metadata.json may also exist for
debugging, but they are not scientific input to tables or figures.

## Automatic aggregation

The pipeline:

1. discovers canonical run directories;
2. excludes explicitly failed/partial runs;
3. fails closed if a completed run is corrupt or missing its declared primary
   metric;
4. records SHA-256 hashes for canonical raw artifacts;
5. flattens scalar summary/resource/task metrics into all_runs.csv;
6. groups seed replicates by suite, job, and declared primary metric;
7. computes deterministic nonparametric bootstrap confidence intervals;
8. writes CSV and Markdown paper tables;
9. generates a primary-metric figure with confidence intervals;
10. indexes and hashes every suite_manifest.json;
11. writes artifact_report.json with the exact aggregation settings.

No metric is manually transcribed into a table or figure, and no missing metric
is silently replaced with a different statistic.

The generic all-runs table retains every scalar summary metric plus task-level
and resource metrics, so later paper-specific analyses can be regenerated from
the same provenance-checked evidence.
