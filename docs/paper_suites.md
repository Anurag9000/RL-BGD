# Curated paper experiment suites

Phase 14 provides a single registry and launcher for the paper-oriented
experiment portfolio.

List of registered suites:

- smoke
- dev
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

Dry-run manifest generation:

    python scripts/run_paper_suite.py smoke --output-root artifacts/suites

Execution:

    python scripts/run_paper_suite.py smoke --output-root artifacts/suites --execute

The launcher expands seeds into immutable run IDs and writes:

- suite_manifest.json;
- one run directory per expanded job;
- stdout.json;
- stderr.log;
- run_metadata.json with git commit, timestamps, duration, return code, and
  complete scientific metadata;
- suite_execution_summary.json.

Large benchmark suites are intentionally not executed by routine CI. Their
manifests are validated statically against the actual Python callable
signatures, including optional CARL and Continual World targets. This catches
stale/nonexistent runner references without pretending the expensive
experiments were executed.
