# Experiments

Implementation proceeds from mathematical mechanism tests to stationary RL,
controlled nonstationarity, recurrent task-agnostic agents, and finally large
external benchmarks. Passing imports is never empirical evidence, and a
runnable benchmark path is not labelled as an executed result without raw run
artifacts.

## Current evidence levels

Synthetic and controlled tests currently exercise BGD posterior mechanics,
tempering, replay-evidence accounting, surprise-driven retention, feed-forward
SAC/PPO, recurrent PPO, recurrent SAC, and recurring hidden-context LQR streams.
Stationary learning acceptance exists for the core feed-forward SAC/PPO paths.

Canonical task-aware Continual World and strict TA-CW10/TA-CW20 now have
runnable Meta-World adapters, training runners, evaluation matrices, configs,
RNG isolation, and simulator cleanup. The full one-million-step-per-occurrence
CW runs remain **not executed** in the repository evidence registry.

CARL's strict hidden-context adapter is now exercised against the real pinned
carl-bench 1.1.1 package in an isolated workflow. The live Pendulum smoke
validates reset/step behavior, scheduled context changes, and context stripping
without importing CARL's legacy Gymnasium constraints into the base environment.

EWC, Online EWC, Synaptic Intelligence, and MAS now have actor/critic SAC
integration. The default comparison uses update-count-driven consolidation with
no task ID or boundary input; a separate oracle-boundary trainer exists and is
labelled as privileged. Both share the same regularizer implementation.

Recurrent BGD-SAC and recurrent BGD-PPO have both passed stationary learning
acceptance under matched recurrent controls. The recurrent PPO acceptance was
reworked after the first naturally stable LQR criterion proved uninformative;
the current gate uses a control-requiring stationary LQR with a matched
recurrent-Adam comparison rather than accepting a trivial stable policy.

ContinualBench now has a strict hidden-task adapter matching its pinned source
contract, including reward dictionaries and both four-/five-value step APIs.
The pinned live reset/step workflow passes. Compatibility handling is confined
to the adapter boundary: reachable missing assets are restored from canonical
Meta-World paths, an upstream debug-only undefined symbol is supplied without
changing reward computation, and the pinned no-op close stub is tolerated.

CORA now has dependency-free canonical protocol metadata plus isolated
forgetting and isolated zero-shot forward-transfer metrics with tests. Separate
legacy workflows pin CORA revision
`f2754bb282757829765beb4703f24b87efa13ff9` and live-validate real upstream
Atari, Procgen, and NetHack/MiniHack task construction plus reset/step behavior
without installing those historical dependencies into RL-BGD's modern runtime.
The MiniHack smoke bridges only CORA's obsolete private `_vardir`/seed wrapper
assumptions.

CHORES/ALFRED has a pinned Xvfb/`crl_alfred` manual recovery workflow, a
complete archive validator derived from CORA's four pinned metadata files, and
an exact published-demo smoke. The validator requires all 27 train/valid_seen
trajectory references and every raw goal image before runtime. Faithful
execution remains blocked only by the unavailable official regenerated-
trajectory archive; generic ALFRED or ALFWorld trajectories are not treated as
substitutes.


UCL-PPO is implemented as an explicitly oracle-boundary comparator because the
original method snapshots the previous-task posterior at known task boundaries.
FOO-VB Diagonal is represented through an exact equivalence contract with the
eta=1 untempered diagonal BGD update rather than duplicated numerical code.

Phase 13-15 infrastructure now includes a reproducible mechanistic analysis
suite, curated paper experiment manifests/launchers, and automatic raw-run
aggregation with deterministic bootstrap confidence intervals. These
infrastructure paths are distinct from expensive experiment execution: paper
claims remain pending until corresponding raw run artifacts exist.

## Suite execution and resume

Paper suites are materialized without execution by default:

    python scripts/run_paper_suite.py <suite> --output-root artifacts/suites

Execute a suite with:

    python scripts/run_paper_suite.py <suite> --output-root artifacts/suites --execute

Execution is resumable by default. A run is skipped only when its saved
`run_metadata.json` reports a strict successful result, its full execution
contract matches the newly materialized job, its git commit matches the suite
manifest, and the canonical run artifacts can be loaded successfully. Failed,
partial, corrupted, stale-commit, or contract-mismatched runs are executed
again automatically.

Use `--continue-on-error` to preserve a failed run artifact and continue with
the remaining jobs. Use `--no-resume` to force every declared job to execute
again even when a matching strict success already exists. The suite execution
summary records selected, executed, skipped, and failed run IDs explicitly.

Large suites can be partitioned deterministically for cluster, Slurm, or
multi-GPU orchestration without weakening provenance. Repeat `--job-id`,
`--seed`, or `--run-id` to select an intersection of declared jobs. For
example:

    python scripts/run_paper_suite.py cw20_final \
      --output-root artifacts/suites \
      --execute \
      --job-id cw20_ta_bgd \
      --seed 3

Materialization still writes the complete suite manifest. A filtered worker
executes only its selected run directories and writes a selection-specific
summary under `execution_summaries/`, so independent workers do not overwrite
the full-suite execution summary. The suite-manifest write is atomic, allowing
separate workers to materialize the same suite safely. Unknown job IDs, seeds,
or expanded run IDs fail before training starts.

For GPU arrays, launch one filtered process per assigned GPU/process and leave
the job's `device: auto` setting intact; process-level CUDA visibility can be
set by the scheduler. This avoids hidden in-process GPU contention while
preserving the same runner code and canonical artifacts.

Superseded or forced-rerun results are never overwritten in place. Before a
rerun starts, an existing run directory is moved intact to the sibling
`<output_root>_archives/<suite>/<run_id>/...` tree. The replacement run's
metadata records the archive path and the execution summary lists archived
runs. The archive tree remains outside the active results root, so ordinary
paper aggregation sees only current canonical runs unless an archive is
explicitly selected as an input.
