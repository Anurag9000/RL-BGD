# Adaptive Surprise-Driven Replasticization

RL-BGD supports task-boundary-free posterior tempering driven by online surprise.

For TD surprise, the agent computes Bellman residuals from the current replay batch using the current mean critics and the ordinary SAC target. The residual magnitude is aggregated, normalized against exponential running center/variance statistics, and smoothed online. No task ID, task index, switch time, or boundary callback is consumed.

The implemented monotone retention mapping is

    lambda_t = lambda_min + (1 - lambda_min) * exp(-kappa * Sbar_t)

so zero surprise yields lambda=1 and increasing surprise approaches lambda_min.

The resulting retention value is supplied as a per-update override to BGDUpdater. Tempering occurs before the corresponding Bayesian evidence update. If adaptive retention is disabled, each BGD updater falls back to its configured fixed temper_retention.

RL-BGD currently exposes three mutually exclusive adaptive surprise sources in BGD-SAC:

- TD/Bellman residual surprise from the current replay batch;
- twin-critic ensemble disagreement;
- predictive negative log likelihood from an online Gaussian transition-and-reward model.

For predictive surprise, the world model scores each replay batch **before** it is trained on that batch, so the surprise signal is not artificially reduced by fitting the observation first. The model is then updated by maximum likelihood. Its parameters, optimizer state, and surprise normalizer are included in the BGD-SAC checkpoint. Only one adaptive source may control retention in a run, which keeps causal comparisons identifiable.

Evaluation may compare detected surprise events to known environment changes. Training code must not receive those change labels.

## Recurrent BGD-SAC adaptive-surprise contract

The sequence-replay BGD-SAC agent supports the same four mutually exclusive
retention modes: fixed, TD surprise, twin-critic ensemble disagreement, and
predictive Gaussian transition/reward negative log likelihood. Each surprise
measurement precedes the corresponding Bayesian actor/critic update.

TD and ensemble statistics use critic predictions only from the optimized
unroll positions. Burn-in observations may reconstruct the recurrent hidden
state but are not independently counted as surprise/evidence observations.
Predictive NLL scores a flattened batch of the optimized unroll transitions
with the same agent-visible observation vector as the recurrent models,
including previous-transition context when configured. The predictive model
is fitted **after** scoring surprise and performing the SAC updates, using
those unroll positions only.

All adaptive modes use the existing EMA normalizer and monotone retention
mapping. Diagnostic keys include `surprise_raw`, `surprise_center`,
`surprise_scale`, `surprise_normalized`, `surprise_smoothed`, and
`retention_lambda`; predictive mode additionally emits
`predictive_model_loss`. The recurrence-aware BGD checkpoint schema is
version 3 and includes selected estimator state plus, for predictive mode,
the world-model and optimizer states. Version 2 checkpoints remain readable
for TD-only and fixed-retention configurations. Missing required estimator
state and conflicting configurations are errors, not silent resets.

These are implementation semantics, not evidence of completed CW10/CW20
training or confirmation of any scientific performance hypothesis.
