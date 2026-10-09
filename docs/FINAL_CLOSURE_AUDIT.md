# Final Repository Closure Audit

This document records the implementation-closure state of RL-BGD after the
Phase-16 repository audit. It deliberately separates software completeness from
scientific execution and scientific conclusions.

## Closure status

At commit `c51de352c6ffc1f04209229ef36275257ffb1cdc`, the required modern-runtime
validation gates all passed:

- Ruff rule checks and Ruff formatting;
- mypy over `src/rl_bgd`;
- the full CPU test suite selected by the main test workflow;
- live CARL adapter/training smoke;
- end-to-end paper-suite -> canonical raw artifacts -> paper-artifact smoke.

The repository also has passing isolated validation for:

- the pinned ContinualBench runtime after its documented upstream compatibility
  shim;
- the pinned CORA Atari runtime;
- the pinned CORA Procgen runtime;
- the pinned CORA MiniHack runtime;
- recurrent BGD learning acceptance for recurrent PPO and recurrent SAC.

The repository closure test is part of the normal CPU test suite and validates
the required repository surfaces, experiment registry, source/config
references, forbidden unfinished markers, duplicate top-level definitions, and
canonical artifact authority.

## Repository hygiene

The closure audit verified:

- `main` is the only branch;
- there are no open pull requests;
- no `TODO`, `FIXME`, or `NotImplementedError` markers remain in the
  audited source surfaces;
- no hard-coded `.cuda()` calls remain;
- validation workflows use concurrency cancellation so obsolete runs no longer
  accumulate ahead of the newest direct-to-main commit;
- paper runs use suite-namespaced run IDs;
- superseded run directories are archived rather than overwritten;
- filtered suite execution writes selection-specific execution summaries;
- paper aggregation consumes only canonical completed run artifacts.

## Implementation-complete research surfaces

The capability ledger has no PARTIAL or NOT STARTED entries. Implemented and
tested surfaces include:

- diagonal-Gaussian BGD and generalized-Bayes evidence temperature;
- replay-evidence accounting and uncertainty diagnostics;
- SAC, PPO, BGD-SAC, BGD-PPO, recurrent SAC, recurrent PPO, and recurrent BGD
  variants;
- fixed and adaptive posterior tempering;
- TD, ensemble-disagreement, and predictive-NLL surprise sources;
- task-aware and strict task-agnostic Continual World protocols;
- 3RL-style recurrent Continual World;
- strict hidden-context CARL;
- ContinualBench;
- CORA protocol/metric compatibility and isolated Atari/Procgen/MiniHack
  runtimes;
- EWC, Online-EWC, SI, MAS, UCL-PPO, and the diagonal FOO-VB equivalence
  baseline;
- mechanistic movement, perturbation, freezing, curvature, and uncertainty
  analyses;
- curated resumable/partitionable paper suites;
- canonical raw-run schemas with provenance hashes;
- automatic bootstrap statistics, matched-seed comparisons, tables, and
  figures.

## Incremental reproducibility closure (October 2026)

Subsequent to the historical Phase-16 validation snapshot above, incremental
commits added exact SAC/PPO and recurrent SAC/PPO training resume, stationary
command-line checkpoint controls, replay/hidden/environment RNG restoration,
and fail-closed scientific checkpoint configuration checks. Regression tests
compare resumed and uninterrupted runs on dependency-light synthetic LQR
without claiming any final multi-seed scientific results.

The audit also validates SAC/replay and PPO/rollout progress alignment,
including rejection of non-finite saved replay transitions and rollout values,
non-negative replay insertion provenance, transactional staging of replay
payloads before live-buffer mutation, and transactional validation of complete
live replay transitions before insertion. Wrapped recurrent replay validates
episode boundaries as well. Checkpoints enforce strict integer and recurrent
episode-boundary flags, strict integer version metadata, exact restored
observation shapes in SAC/PPO and recurrent variants, and transactional process
RNG restore across Python, NumPy, PyTorch CPU, and CUDA state. RNG restore also
fails closed on incompatible CPU/CUDA checkpoint-runtime topology rather than
silently leaving a CUDA stream unrestored. PPO rollout checkpoint rejection is
non-mutating. Rollout restore
also rejects non-finite saved observations, actions, rewards, policy
statistics, GAE estimates, and recurrent hidden states, and stages all validated
payload tensors on the destination device before mutating live rollout state. Recurrent rollout
restore additionally verifies that interior episode-start flags agree with
preceding terminal/truncation boundaries. Both PPO rollout variants reject
computed GAE payloads for an empty buffer. These have focused
corruption/round-trip tests alongside split-vs-uninterrupted runs.

The audit also validates schedule mode/step inputs, strict seed and context
types, finite context parameters, random-walk anchor cardinality, mode-specific
bounds usage, and bounded-walk starting states. It guards posterior
bounds/precision and surprise normalization settings against silent checkpoint
drift. Scheduled-LQR restore cross-checks
saved context against both the base simulator parameters and the context implied
by saved stream progress. The strict CARL adapter also synchronizes each hidden
scheduled context through CARL's reset-time selector before the wrapped
environment resets, preventing stale-context episode initialization. Version-2
surprise checkpoints intentionally reject incompatible earlier payloads rather
than silently guessing their unrecorded configuration. Training checkpoints are
published through unique same-directory staging files; data are fsynced before
atomic replacement and POSIX parent directories are fsynced afterward so a
completed publication is crash-durable at the directory-entry level.

The opening commit-specific CI statement above is a historical evidence
snapshot, not a claim that every later commit has already passed all gates.
The current main-branch workflows remain the authoritative evidence for
subsequent changes. The unresolved CHORES external archive prerequisite and
the unexecuted scientific suites below are unchanged.

## External block: CORA CHORES / ALFRED

CORA CHORES is the only capability still marked BLOCKED.

The implementation side is complete: a pinned isolated Xvfb/crl_alfred runtime,
secure official-source/mirror downloader, bounded invalid-archive diagnostics,
collision-free temporary download staging, atomic archive publication, optional
SHA-256 verification, safe ZIP extraction, metadata closure checks, complete
27-trajectory/raw-image validation, and the exact published trajectory smoke
path are present. The recovery workflow defaults to CORA's historical OneDrive
URL and can accept an authoritative replacement mirror without code changes.
Execution is blocked because the certified 2021 CORA CHORES archive is no
longer retrievable from the original source and no authoritative replacement
has been recovered.

The repository must not substitute regenerated or guessed trajectories while
claiming reproduction of the published CORA CHORES result.

## Scientific execution still pending

Implementation completeness does not imply that the paper hypotheses are
confirmed. The following remain experiment-execution work:

- five-seed stationary Adam/BGD controls;
- baseline-core EWC/Online-EWC/SI/MAS/UCL comparisons;
- CARL abrupt/smooth/recurring studies;
- canonical and task-agnostic CW10;
- canonical and task-agnostic CW20;
- recurrent CW10/CW20 comparisons;
- replay/evidence-temperature/Bayesianization/Monte-Carlo ablations;
- uncertainty-source and controlled-forgetting studies;
- multi-seed mechanistic artifact generation;
- compute-cost measurements.

No expensive experiment should be marked executed until its canonical raw run
directory exists and passes the strict artifact loader.

## Canonical execution path

Materialize a suite without running it:

    python scripts/run_paper_suite.py smoke --output-root artifacts/suites

Execute a suite:

    python scripts/run_paper_suite.py smoke --output-root artifacts/suites --execute

Large suites can be partitioned by job ID, seed, or expanded run ID for
independent GPU/cluster workers. Successful runs write the canonical raw-run
contract:

- `manifest.json`
- `config.yaml`
- `metrics.csv`
- `summary.json`

Then build paper artifacts only from the canonical results tree:

    python scripts/build_paper_artifacts.py \
        --results-root artifacts/suites \
        --output-dir artifacts/paper

The paper builder fails closed on failed/partial/corrupt runs, duplicate
run/seed identities, inconsistent matched-seed metrics, or missing declared
primary metrics.

## Final interpretation

Software implementation closure: **complete**, except for the externally
blocked CORA CHORES data dependency.

Scientific execution: **pending** for the expensive multi-seed suites listed in
`docs/EXPERIMENT_REGISTRY.md`.

Scientific claims: **not yet established** until those canonical run artifacts
are generated and aggregated.
