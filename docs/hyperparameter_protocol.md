# Hyperparameter Protocol

Development/tuning environments and final held-out evaluation must remain separate. Every sweep records search space, budget, seeds, selection metric, and selected configuration. Baselines receive comparable tuning budgets. Final benchmark aggregates are not repeatedly reused for tuning.

The initial mathematical phase uses deterministic fixed parameters solely to verify equations and numerical behavior; these are tests, not benchmark tuning.
