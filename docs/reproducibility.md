# Reproducibility

RL-BGD separates **runnable experiment definitions** from **executed evidence**.
A benchmark or algorithm is not treated as empirically validated merely because
its imports or launch path work.

## Deterministic controls

The shared seeding utility seeds Python, NumPy, PyTorch CPU, and CUDA RNGs and
supports deterministic debug behavior. Environment, replay, task-stream, and
evaluation seeds are passed explicitly by runners. Bitwise equality across
different GPU architectures, CUDA versions, or third-party simulators is not
promised.

## Canonical raw-run artifacts

Completed Phase-14 suite jobs are converted automatically into a canonical run
directory containing:

- `manifest.json`: run ID, method, benchmark, protocol, seed, git revision,
  task order, information-access assumptions, status, and scientific metadata;
- `config.yaml`: the runner target, user-supplied kwargs, fully resolved
  call arguments after Python signature defaults are applied, plus any referenced
  configuration template;
- `metrics.csv`: raw timeline/matrix rows or scalar result rows;
- `summary.json`: finite scalar metrics, per-task metrics, and resources;
- launcher stdout/stderr and execution metadata where the suite runner is used.

The fully resolved call arguments are the authoritative call-level execution
contract. Referenced YAML is retained for traceability and human comparison, but
it is not allowed to override what the runner actually received. Scientific
settings that were previously hard-coded in the stationary SAC/PPO runners are
now explicit function arguments so they appear in this resolved contract.

The loader validates schema versions, finite scalar values, run-ID consistency,
artifact presence, and completion status. Every loaded source file receives a
SHA-256 digest in the generated paper provenance manifest. Failed runs receive
a failed manifest and are rejected by the strict paper builder rather than
silently disappearing from a seed aggregate.

## Checkpointing

Bayesian posterior/updater state, SAC/PPO model and optimizer state, replay
state where applicable, recurrent hidden/rollout state, surprise normalization,
and predictive-world-model state use versioned checkpoint paths with round-trip
tests in their owning components. Resume semantics are tested at the component
level; exact cross-hardware floating-point identity is not claimed.

Stationary and recurrent replay checkpoint loaders validate integer metadata,
tensor shapes/dtypes, physical ring chronology, transition IDs, and evidence
usage/freshness before copying any tensors into the live buffer. Rejected
payloads therefore leave existing replay state intact; wrapped-ring round-trip
regression tests compare subsequent seeded samples after restoration.

## Resumable stationary training

The dependency-light stationary SAC/PPO and BGD-SAC/BGD-PPO script entrypoints
expose checkpoint, resume, and bounded-per-call controls. Recurrent stationary
acceptance runs have a single-run CLI so a worker never has to launch all
recurrent acceptance experiments merely to resume one checkpoint.

Examples:

    python scripts/run_sac_lqr.py \
        --checkpoint-path artifacts/checkpoints/sac.pt \
        --checkpoint-interval 100

    python scripts/run_sac_lqr.py \
        --resume-from artifacts/checkpoints/sac.pt

    python scripts/run_bgd_ppo_lqr.py \
        --checkpoint-path artifacts/checkpoints/bgd_ppo.pt \
        --checkpoint-interval-rollouts 1 \
        --max-rollouts-this-call 2

    python scripts/run_recurrent_stationary_lqr.py \
        --algorithm bgd_sac \
        --checkpoint-path artifacts/checkpoints/recurrent_bgd_sac.pt \
        --checkpoint-interval 100

Training checkpoints bind the saved learner state to the complete training and
agent configurations. SAC checkpoints also preserve replay contents,
provenance/evidence-use metadata, replay-sampling RNG, environment state, and
process RNG state. PPO checkpoints are written only at rollout boundaries;
recurrent variants also preserve online hidden state and recurrent progress.
Configuration mismatches fail closed on restore. Saved SAC progress must agree
with the replay transition history; PPO progress must agree with completed
rollout updates. Saved counters and recurrent episode-boundary flags are
validated without implicit numerical or boolean coercion.

Feed-forward and recurrent PPO rollout buffers validate checkpoint dimensions,
tensor shapes/dtypes, and paired GAE advantages/returns before updating live
on-policy buffers. Rejected corrupt rollout payloads do not partially overwrite
existing behavior-policy statistics.

Version-2 surprise-normalizer, TD-surprise and previous-transition
context-wrapper checkpoints deliberately reject older payload formats that
lacked configuration fields. Earlier checkpoints must be regenerated or
migrated with independently verified original settings; guessing defaults
would silently alter the experiment. Checkpoint files use PyTorch
deserialization and should only be loaded from trusted local sources.

Environment restore also rejects nonfinite mutable dynamics and inconsistent
scheduled-context state. Nonstationary schedule constructors reject unknown
modes and nonfinite parameterization before any training begins.

## Paper statistics

Paper aggregation uses matched seed groups. Scalar metrics receive percentile
bootstrap confidence intervals; paired method comparisons resample matched seed
differences; task-within-seed measurements use hierarchical seed-then-task
bootstrap instead of pretending correlated task scores are independent.

Suite revision 3 makes pairwise comparison opt-in through a declared
`comparison_group` and records fully resolved runner-call defaults as part of
the execution contract. Before pairing, Phase 15 requires the same suite,
comparison group, protocol, benchmark, seed set, primary outcome, suite
revision, git revision, evaluator-owned information privileges, and task order.
Jobs without a comparison group are aggregate-only. Raw run IDs, comparison
groups, and suite revisions remain traceable through generated tables.

## Experiment execution policy

Curated `smoke`, `dev`, CARL, CW10, CW20, task-agnostic, ablation,
uncertainty, mechanism, and compute suites declare targets, kwargs, seeds,
protocols, source configs, primary/secondary metrics, dependency extras, and
runtime classes. Expensive final runs stay explicitly unexecuted until their
canonical raw directories exist.

Development/tuning and final evaluation must remain separate. Failed seeds,
changed task orders, or altered final protocols may not be silently removed
after results are observed.
