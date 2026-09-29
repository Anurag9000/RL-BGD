# Architecture

## Design rule

Research semantics are separated from RL training loops. Posterior representation, BGD updates, tempering, evidence accounting, surprise estimation, agents, benchmark streams, evaluation, logging, and paper artifacts must remain independently testable.

## Current package layers

- `rl_bgd.bayes`: posterior representation, MC sampling, BGD update, tempering, diagnostics, generalized-Bayes metadata.
- `rl_bgd.continual`: information-access contracts and later stream/regularization abstractions.
- `rl_bgd.utils`: device and reproducibility controls.
- `rl_bgd.runners`: runnable smoke/research entry points.

Planned layers are intentionally not represented by fake implementations. They remain PARTIAL or NOT STARTED in the capability ledger until source, wiring, tests, and runnable paths exist.

## Bayesian update boundary

`BGDUpdater.step(objective)` receives a differentiable objective over a sampled parameter mapping. SAC/PPO will define objective closures rather than BGD importing RL-specific logic. `step_module` uses `torch.func.functional_call` for simple module objectives.

## Numerical policy

Posterior states and MC statistics are FP32; sampled weights use the model computation dtype. Sigma is bounded and non-finite state fails loudly. Device auto-selection is CUDA-first. CuPy is not introduced because the numerical path is already GPU-native PyTorch.

## Information-access boundary

`InformationAccessConfig` rejects task IDs, task-boundary notifications, task-specific heads, task-balanced replay, and exposed ground-truth context in strict task-agnostic mode.
