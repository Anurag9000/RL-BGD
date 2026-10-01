# Paper artifact pipeline

Phase 15 consumes only raw outputs created by the Phase 14 suite launcher.

Run:

    python scripts/build_paper_artifacts.py \
        --run-root artifacts/suites \
        --output-dir artifacts/paper

The pipeline automatically:

1. discovers successful run_metadata.json + stdout.json pairs;
2. records SHA-256 provenance for raw outputs and metadata;
3. flattens numeric result fields into all_runs.csv;
4. resolves each job's declared primary metric;
5. groups seed replicates by suite/job;
6. computes deterministic nonparametric bootstrap confidence intervals;
7. writes CSV and Markdown paper tables;
8. generates a primary-metric figure with confidence intervals;
9. indexes and hashes every suite_manifest.json;
10. writes artifact_report.json with the exact aggregation settings.

No metric is manually transcribed into a table or figure. Missing primary metrics
remain missing instead of being silently invented or substituted.

The generic all-runs table retains every numeric result field, so later
paper-specific figures can be regenerated from the same raw evidence.
