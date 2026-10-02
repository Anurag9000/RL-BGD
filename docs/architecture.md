# Architecture

## Design rule

Research semantics are separated from RL training loops. Posterior
representation, Bayesian updates, controlled forgetting, evidence accounting,
surprise estimation, agent algorithms, benchmark streams, evaluation,
experiment orchestration, and paper artifacts remain independently testable.

## Package layers

- `rl_bgd.bayes`: diagonal Gaussian posterior state, Monte Carlo sampling,
  BGD/FOO-VB-equivalent updates, exact Gaussian tempering, generalized-Bayes
  evidence temperature, and posterior diagnostics.
- `rl_bgd.surprise`: online normalization plus TD, ensemble-disagreement, and
  predictive-NLL surprise sources.
- `rl_bgd.replay`: transition and sequence replay, usage/freshness metadata,
  and explicit Bayesian evidence-accounting modes.
- `rl_bgd.models`: feed-forward/recurrent SAC and PPO networks, task-aware
  Continual World heads, and UCL Bayesian layers.
- `rl_bgd.agents.sac`: Adam SAC, BGD-SAC, recurrent SAC/BGD-SAC,
  task-aware canonical SAC, and EWC/Online-EWC/SI/MAS-regularized variants.
- `rl_bgd.agents.ppo`: Adam/BGD PPO, recurrent Adam/BGD PPO, rollout
  checkpointing/evidence modes, and oracle-boundary UCL-PPO.
- `rl_bgd.continual`: information-access contracts, context schedules, and
  continual-learning regularization abstractions.
- `rl_bgd.envs`: synthetic controlled environments, hidden-context wrappers,
  CARL, modern Meta-World/Continual World, ContinualBench, and protocol types.
- `rl_bgd.metrics`: continual, adaptation/recovery, change-detection, and CORA
  compatibility metrics.
- `rl_bgd.runners`: reproducible stationary, continual, recurrent, benchmark,
  ablation, and mechanism entry points.
- `rl_bgd.experiments`: curated Phase-14 paper suite registry, static
  signature/config validation, and launcher.
- `rl_bgd.artifacts`: canonical raw-run schema, strict loading, provenance,
  runner-result conversion, and suite/run bridge.
- `rl_bgd.analysis`: mechanistic analyses, bootstrap statistics, strict
  Phase-15 paper aggregation, tables, and figures.
- `rl_bgd.audit`: executable Phase-16 repository closure checks.
- `rl_bgd.utils`: CUDA-first device selection and deterministic seeding.

Optional CARL/Continual World/ContinualBench dependency stacks are isolated from
the base install so legacy or simulator-specific pins do not silently redefine
the primary runtime.

## Bayesian update boundary

`BGDUpdater.step(objective)` receives a differentiable objective over sampled
parameter mappings and contains no SAC/PPO-specific logic. RL agents construct
objective closures around their own surrogate losses.

`BGDLoss` separates posterior-mean learning from uncertainty evidence. This
keeps optimization and consolidation semantics explicit: ordinary replay can
continue training the mean while evidence weights modify only the sigma-driving
gradient channel.

For module-local objectives, `step_module` uses
`torch.func.functional_call`. Actor/critic implementations use the same
functional style where sampled parameters must be evaluated without mutating
the live mean module.

## Posterior-state lifecycle

The live PyTorch module is synchronized to the posterior mean after a Bayesian
update. SAC target critics track these means using ordinary Polyak averaging;
the default target networks do not maintain separate posterior uncertainty.

Exact tempering may alter the posterior before a BGD evidence step. Adaptive
retention obtains its per-update lambda from learner-observable surprise only.
Posterior, surprise, replay, optimizer, recurrent, and predictive-model state
have explicit checkpoint owners rather than one implicit global singleton.

## Environment and information boundary

Tensor-native training loops depend on shared structural protocols rather than
specific benchmark classes. Strict task-agnostic adapters expose ordinary
interaction data while withholding task ID, task boundary, simulator context,
and evaluator labels.

Evaluator-owned task/stage knowledge is isolated in separate environments or
bookkeeping objects and may construct matrices/change-detection labels without
becoming an agent input.

Canonical task-aware and oracle-boundary comparisons are separate protocol
families and store their information-access assumptions in run manifests.

## Experiment and artifact boundary

Phase-14 suites describe what should run: callable, kwargs, seed set, protocol,
source config, metrics, dependencies, and runtime class. Successful invocations
are converted into the canonical raw-run schema:

    manifest.json
    config.yaml
    metrics.csv
    summary.json

Phase-15 aggregation consumes only these canonical artifacts through the strict
loader; stdout and execution logs are debugging surfaces, not paper data.
Failed/partial manifests make strict aggregation fail closed.

Phase-16 audits structural closure, suite references, duplicate definitions,
forbidden source placeholders, canonical artifact authority, and required paper
surfaces.

## Numerical policy

Posterior means/stds and Monte Carlo BGD statistics are FP32. Standard
deviations have positive finite bounds and non-finite Bayesian state fails
loudly. Device selection is CUDA-first with CPU fallback.

The numerical training path is PyTorch-native on the selected device. CuPy is
not inserted into tensor training paths merely to satisfy a GPU-library
preference, because transferring between PyTorch and CuPy would add another
memory/runtime boundary without improving the model computation path.

## Evidence levels

Implementation, wiring/tests, live optional-dependency smoke, and full
paper-scale execution are distinct evidence levels. A runnable CW20 path is not
an executed CW20 result; a passing import is not a benchmark validation; and an
implemented figure pipeline does not create data that were never measured.
