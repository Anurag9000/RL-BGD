# Automatic paper artifact pipeline

Phase 15 is generated from raw per-run directories; paper numbers are never
manually transcribed.

A completed run contains:

- `manifest.json`: run ID, method, setting, benchmark, seed, git commit,
  task order, information-access assumptions, and scientific metadata;
- `config.yaml`: the resolved invocation and source configuration;
- `metrics.csv`: timeline rows or a scalar-result row;
- `summary.json`: finite scalar metrics, per-task metrics, and resource
  measurements.

The Phase-14 suite launcher writes this schema automatically after each
successful JSON-producing job. Process failures or artifact-conversion failures
receive a failed manifest; the paper builder refuses to aggregate them by
default, so failed seeds cannot disappear silently.

Build all paper artifacts with:

    python scripts/build_paper_artifacts.py \
      --results-root results \
      --output-dir artifacts/paper

The builder performs seed-level percentile bootstrap confidence intervals,
matched-seed paired method differences, and hierarchical seed-then-task
bootstrap intervals for task-level measurements. It rejects duplicate run IDs,
duplicate seeds inside one experiment/method/setting/benchmark group,
information-access inconsistencies, and metrics missing from only a subset of
matched seeds.

Generated tables include:

- aggregate scalar statistics;
- hierarchical task statistics;
- paired method differences;
- a human-facing results summary;
- information-access assumptions;
- a run/provenance index.

Every table is exported as CSV, Markdown, and LaTeX. The generated
`paper_manifest.json` records every source run path, git commit, source-file
SHA-256 hash, statistical configuration, generated artifact, and skipped figure
with its reason.

Figures are created only when their required raw metrics exist. Missing
evidence produces an explicit skipped-figure entry rather than an invented
value. Large CW10/CW20 paper runs remain not executed until raw run directories
exist; the artifact pipeline does not convert runnable code into an empirical
claim.
