# Reproducibility

RL-BGD separates **runnable experiment definitions** from **executed evidence**.
A benchmark or algorithm is not treated as empirically validated merely because
its imports or launch path work.

## Deterministic controls

The shared seeding utility seeds Python, NumPy, PyTorch CPU, and CUDA RNGs and
supports deterministic debug behavior. Seeds are strict non-negative integers;
booleans and numerically coercible non-integers are rejected. Environment,
replay, task-stream, and evaluation seeds are passed explicitly by runners.
Exact process-RNG restore also fails closed when checkpoint and runtime CUDA
availability/device topology differ rather than silently leaving a CUDA stream
unrestored. Bitwise equality across different GPU architectures, CUDA versions,
or third-party simulators is not promised.

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
artifact presence, and completion status. Artifact filenames must be distinct
from one another and from `manifest.json`; both writing and reading reject
paths that resolve outside the run directory, including symlink escapes.
The writer rejects blank or non-string metric-column names, and the loader
checks raw CSV headers for duplicates before pandas could rename them.
Raw-run publication stages configuration, summary, metrics, and manifest bytes in
temporary files. The payloads are published and synchronized before
`manifest.json` is published last as the completion marker. Failed
serialization leaves no discoverable completed run, and an existing published
manifest cannot be overwritten: rerun with a fresh run directory or archive the
old directory first. POSIX directory entries are fsynced before completion
publication. Simultaneous writers to the same previously empty run ID are not
a supported concurrency mode; workers must use unique run IDs.
Every loaded source file receives a SHA-256 digest in the generated paper
provenance manifest. Failed runs receive a failed manifest and are rejected by
the strict paper builder rather than silently disappearing from a seed aggregate.

## Checkpointing

Bayesian posterior/updater state, SAC/PPO model and optimizer state, replay
state where applicable, recurrent hidden/rollout state, surprise normalization,
and predictive-world-model state use versioned checkpoint paths with round-trip
tests in their owning components. Resume semantics are tested at the component
level; exact cross-hardware floating-point identity is not claimed.

Stationary and recurrent replay checkpoint loaders validate integer metadata,
tensor shapes/dtypes, physical ring chronology, transition IDs, non-negative
insertion-step provenance, recurrent episode-start boundaries, and evidence
usage/freshness before copying any tensors into the live buffer. Live replay
insertion uses the same fail-closed contract: observations/actions must have
exact vector shapes and finite floating values, rewards must be finite real
scalars, boundary flags must be booleans, and insertion steps must be
non-negative integers. Sequence replay additionally checks that each new
episode-start flag agrees with the preceding terminal or truncation flag,
including after ring wrap; the first resident insertion may start mid-episode.
Each complete transition is staged before any replay slot is mutated. Validated checkpoint payload tensors are likewise staged
completely on the destination device before live storage is touched, so
transfer/conversion failures are transactional. Rejected payloads therefore
leave existing replay state intact; wrapped-ring round-trip regression tests
compare subsequent seeded samples after restoration.

## Resumable stationary training

The dependency-light stationary SAC/PPO and BGD-SAC/BGD-PPO script entrypoints
expose checkpoint, resume, and bounded-per-call controls. Recurrent stationary
acceptance runs have a single-run CLI so a worker never has to launch all
recurrent acceptance experiments merely to resume one checkpoint. The
oracle-boundary regularized SAC trainer also preserves its global replay,
phase-local replay, both sampling RNG streams, environment state, process RNG,
consolidation history, and learner state; its resume validator binds phase
replay progress to the most recent declared consolidation boundary. Canonical
MetaWorld Continual World training deliberately does not claim exact mid-run
resume because the current external simulator adapter does not expose a complete
restorable simulator state.

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

Stationary and recurrent SAC/PPO training checkpoint writers, including the
oracle-boundary regularized SAC variant, use independent same-directory staging
files followed by an atomic replacement. Checkpoint bytes are flushed and
fsynced before publication; on POSIX the parent directory is fsynced after
replacement so the published directory entry is crash-durable. Failed
serialization preserves the preceding complete checkpoint. Multiple writers
targeting the exact same final path
remain last-completed-writer-wins; use distinct run IDs and checkpoint paths
for independent scientific workers.

Training checkpoints bind the saved learner state to the complete training and
agent configurations. SAC checkpoints also preserve replay contents,
provenance/evidence-use metadata, replay-sampling RNG, environment state, and
process RNG state. Process RNG restore validates strict integer version metadata,
Python, NumPy, PyTorch CPU, CUDA payloads, and compatible CUDA runtime topology
before mutating any live global RNG stream, so malformed late fields fail
without perturbing the running experiment. Restored current observations in
SAC/PPO, recurrent SAC/PPO, and boundary-regularized SAC must exactly match the
environment observation shape; recurrent SAC history observations are checked
the same way. Ordinary and recurrent SAC replay restores reject NaN/Inf
transition tensors, negative insertion provenance, and non-integer checkpoint
versions before any live buffer state is modified. PPO checkpoints are written
only at rollout boundaries;
recurrent variants also preserve online hidden state and recurrent progress.
Configuration mismatches fail closed on restore. Saved SAC progress must agree
with the replay transition history; PPO progress must agree with completed
rollout updates. Saved counters and recurrent episode-boundary flags are
validated without implicit numerical or boolean coercion. Continual-learning regularizer checkpoints also reject
non-finite, dtype-coerced, layout-mismatched, or negative-importance tensor
payloads before committing staged state.

Feed-forward and recurrent PPO rollout buffers validate checkpoint dimensions,
strict version metadata, tensor shapes/dtypes, finite floating-point behavior
statistics (including hidden states and GAE results), and paired GAE
advantages/returns before updating live on-policy buffers. NaN/Inf checkpoint
payloads fail closed without partially overwriting behavior-policy statistics.
Validated payload tensors are also fully staged on the destination device before any live
rollout storage is changed, so transfer/conversion failures are transactional.

Version-2 surprise-normalizer, TD-surprise and previous-transition
context-wrapper checkpoints deliberately reject older payload formats that
lacked configuration fields. Earlier checkpoints must be regenerated or
migrated with independently verified original settings; guessing defaults
would silently alter the experiment. Checkpoint files use PyTorch
deserialization and should only be loaded from trusted local sources.

Environment restore also rejects nonfinite mutable dynamics and inconsistent
scheduled-context state. Nonstationary schedule constructors reject unknown
modes, malformed anchor keys/values, non-integer seeds, nonfinite
parameterization, extra random-walk anchors that would be ignored, bounds on
modes that do not consume them, and random-walk starting states outside their
declared bounds before any training begins.

## Bayesian replay evidence validation

Replay uncertainty weighting in both ordinary and recurrent SAC rejects malformed
usage-count tensors, inconsistent freshness flags, and empty replay evidence
before applying **any** weighting mode (including `all_replay`). Weighted
uncertainty-loss reductions reject empty, non-finite, or negative weight inputs
and non-finite losses, preventing silent NaN gradients or invalid evidence
mass from corrupting posterior updates. The weighting formulas and normal
valid-batch behavior remain unchanged; failure paths have focused unit tests.

## Adaptation-metric trace contracts

Adaptation and recurrence metrics reject non-finite scores, reference values,
duplicate or out-of-order evaluation steps. Fixed-window post-change AUC
integrates over the exact declared time window, linearly interpolating scores
at its boundaries from neighboring observed samples. A window extending outside
the observed trace is rejected rather than silently scored over a shortened
duration. Threshold recovery time remains the first *observed* checkpoint that
reaches the target; no unobserved crossing time is inferred.

## Paper statistics

Paper aggregation uses matched seed groups. Scalar metrics receive percentile
bootstrap confidence intervals; paired method comparisons resample matched seed
differences; task-within-seed measurements use hierarchical seed-then-task
bootstrap instead of pretending correlated task scores are independent.
The bootstrap primitives reject boolean/string metric values, non-integer
resample counts and RNG seeds, and non-integer/negative hierarchical seed IDs
rather than silently coercing them and risking seed identity collisions.
Bootstrap estimates also reject non-finite numeric overflow.

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


## Regularizer architecture consistency

EWC consolidations require identical parameter names and shapes across saved
task anchors. Online-EWC and MAS reject changed parameter layouts before
accumulating importance, including changes that would otherwise be silently
broadcast by PyTorch. SI checkpoint restore checks incoming parameter layout
against the live model as well as the other saved SI fields. These checks
reject invalid state before mutating the active regularizer. SI's online
path-integral accumulation and boundary consolidation stage every parameter
before committing state, rejecting invalid later tensors and arithmetic overflow
without leaving earlier parameters partially updated.

## BGD failed-update behavior

BGD updater controls and per-step retention/temperature overrides reject nonfinite
or coerced numeric values. Every posterior update checks scaled objectives,
gradients, aggregated Monte Carlo signals, and the unconstrained updated
mean/standard deviation before accepting the update. If an update fails at
any stage, the updater restores its previous posterior means, standard
deviations, and completed-step count, including changes from adaptive
posterior tempering. This guarantee covers updater state, not user-objective
side effects or consumed random-number-generator draws. It does not imply
that training or benchmark experiments have been executed.

## Posterior-to-module synchronization

Copying Bayesian posterior means to a PyTorch module first checks all target
parameter names, shapes, and dtypes, and stages every cast on its destination
device. A cast that overflows (for example FP32 to FP16) is rejected before
any target parameter is changed. A failed `step_module` synchronization also
rolls back the updater's posterior state and step count. This preflight protects
against validation failures; user-defined model forward side effects and
unexpected device copy failures are outside the transaction guarantee.

## Posterior initialization controls

Posterior standard-deviation bounds and Gaussian prior scalars must be finite
real values, not coerced booleans. Bounds incompatible with the FP32 storage
range are rejected before clamping; this prevents NaN bounds or conversion
underflow from silently producing invalid posterior states.
