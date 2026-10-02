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
