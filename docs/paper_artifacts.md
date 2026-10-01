# Paper artifact pipeline

Phase 15 consumes only canonical raw run directories created by the Phase 14
suite launcher.

Run:

    python scripts/build_paper_artifacts.py \
        --results-root artifacts/suites \
        --output-dir artifacts/paper

## Canonical raw-run contract

Every completed run is authoritative only when its directory contains:

- manifest.json;
- config.yaml;
- metrics.csv;
- summary.json.

The strict loader validates schema versions, run IDs, completion state, source
files, and provenance hashes before a run can enter paper aggregation.
stdout.json, stderr.log, and execution metadata may also exist for debugging,
but they are not scientific inputs to tables or figures.

## Fail-closed aggregation

A paper build does not silently discard a failed or partial run. If the
selected results root contains a failed/partial run manifest, corrupted
completed artifact, duplicate run ID/seed, inconsistent matched-seed metric, or
missing declared primary metric, the build aborts. This prevents survivorship
bias and accidental cherry-picking. Resolve the failed run or intentionally
construct a different, explicitly documented results root before rebuilding.

## Automatic aggregation

The canonical builder automatically:

1. discovers strict run directories;
2. validates run and information-access provenance;
3. aggregates scalar metrics with seed-level bootstrap confidence intervals;
4. hierarchically bootstraps task-level metrics;
5. computes paired matched-seed method differences where available;
6. writes run-index and information-access tables;
7. exports paper tables in CSV, Markdown, and LaTeX;
8. generates publication figures in configured formats;
9. records skipped-figure reasons rather than inventing unavailable data;
10. emits a paper_manifest.json tying every table/figure to source runs.

No metric is manually transcribed and no missing metric is silently substituted.
