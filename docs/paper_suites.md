# Curated paper experiment suites

Phase 14 provides one registry and launcher for the paper-oriented experiment
portfolio.

Registered suites:

- smoke
- stationary_core
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

The `baseline_core` suite is explicitly backbone-matched. SAC-Adam,
SAC-EWC, SAC-Online-EWC, SAC-SI, and SAC-MAS all use the same
`recurring_lqr_matched_v1` stream, 600 environment steps, phase length 120,
horizon 32, network width, optimizer settings, replay budget, and five seeds.
The regularized methods differ only by the continual regularizer and its
fixed-update consolidation state.

PPO-UCL is paired with a phase-matched PPO-Adam oracle control under the same
`oracle_boundary` protocol, stream, phase length, horizon, PPO
hyperparameters, model widths, phases, and five seeds. The Adam control receives
the same evaluator-owned phase segmentation but performs no Bayesian posterior
snapshot or UCL penalty. This lets Phase-15 form matched-seed differences within
each backbone family instead of comparing heterogeneous SAC and PPO conditions.

The `compute_analysis` suite is a matched stationary SAC comparison rather
than a reuse of heterogeneous smoke jobs: SAC-Adam and SAC-BGD
critic-only/actor-only/actor-and-critic each run 600 environment steps on the
same synthetic LQR protocol with seeds 0, 1, and 2. Launcher wall-clock duration
is the primary resource metric and post-training return/improvement are the
matched outcome metrics.


The `cw20_final` suite is comparison-matched rather than method-only. The
feed-forward strict task-agnostic family contains SAC-Adam and SAC-BGD with the
same CW20 stream, one-million steps per task, five evaluation episodes per
stage, and seeds 0-4. The recurrent 3RL-style family contains recurrent
SAC-Adam, recurrent SAC-BGD, and recurrent SAC-adaptive-BGD with the same CW20
stream, one-million steps per task, ten evaluation episodes per stage, recurrent
architecture/config, and seeds 0-4. All task-agnostic CW20 jobs record the same
revisit/reacquisition recurrence metrics; Bayesian jobs additionally expose
posterior uncertainty, and the adaptive job additionally exposes retention.


The `uncertainty_analysis` surprise-source block is also explicitly matched.
A no-adaptation BGD control, TD-residual surprise, twin-critic ensemble
disagreement, and predictive transition/reward NLL all use the same recurring
LQR stream, 900 environment steps, phase length 300, critic-only BGD settings,
EMA normalization hyperparameters, retention mapping, and seeds 0-4. The
no-adaptation arm emits no surprise/retention signal; the adaptive arms share
change-detection metrics, while predictive NLL additionally records its online
model loss.
