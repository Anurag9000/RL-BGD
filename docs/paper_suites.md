# Curated paper experiment suites

Phase 14 provides one registry and launcher for the paper-oriented experiment
portfolio.

Registered suites:

- smoke
- dev
- baseline_core
- carl_core
- cw10_core
- cw20_final
- task_agnostic_final
- ablation_core
- uncertainty_analysis
- mechanism_analysis
- compute_analysis

Each job records its hypothesis ID, callable target, exact kwargs, seeds,
algorithm, environment, information protocol, source config where applicable,
primary/secondary metrics, optional dependency extra, runtime class, and notes.
Hidden-context methods and evidence-temperature values are separate jobs, so
seed-level uncertainty is computed per experimental condition rather than over
a composite JSON blob.

Dry-run manifest generation:

    python scripts/run_paper_suite.py smoke --output-root artifacts/suites

Execution:

    python scripts/run_paper_suite.py smoke --output-root artifacts/suites --execute

Successful jobs are converted into the canonical Phase-15 schema:

- manifest.json
- config.yaml
- metrics.csv
- summary.json

Raw stdout/stderr are retained only for debugging. Failed jobs receive an
explicit failed manifest; downstream paper aggregation refuses such a results
root rather than silently omitting failures.

Large benchmark suites are intentionally not executed by routine CI. The
registry statically validates callable signatures/config references, while a
separate bounded smoke pipeline executes end to end to validate launcher,
canonical artifact conversion, and paper aggregation.
