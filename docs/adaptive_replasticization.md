# Adaptive Surprise-Driven Replasticization

RL-BGD supports task-boundary-free posterior tempering driven by online surprise.

For TD surprise, the agent computes Bellman residuals from the current replay batch using the current mean critics and the ordinary SAC target. The residual magnitude is aggregated, normalized against exponential running center/variance statistics, and smoothed online. No task ID, task index, switch time, or boundary callback is consumed.

The implemented monotone retention mapping is

    lambda_t = lambda_min + (1 - lambda_min) * exp(-kappa * Sbar_t)

so zero surprise yields lambda=1 and increasing surprise approaches lambda_min.

The resulting retention value is supplied as a per-update override to BGDUpdater. Tempering occurs before the corresponding Bayesian evidence update. If adaptive retention is disabled, each BGD updater falls back to its configured fixed temper_retention.

The surprise package also contains ensemble-disagreement and predictive-NLL primitives. They are intentionally not marked as fully integrated adaptive agents until an ensemble/world-model training path supplies those statistics.

Evaluation may compare detected surprise events to known environment changes. Training code must not receive those change labels.
