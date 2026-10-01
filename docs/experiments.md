# Experiments

Implementation proceeds from mathematical mechanism tests to stationary RL,
controlled nonstationarity, recurrent task-agnostic agents, and finally large
external benchmarks. Passing imports is never empirical evidence, and a
runnable benchmark path is not labelled as an executed result without raw run
artifacts.

## Current evidence levels

Synthetic and controlled tests currently exercise BGD posterior mechanics,
tempering, replay-evidence accounting, surprise-driven retention, feed-forward
SAC/PPO, recurrent PPO, recurrent SAC, and recurring hidden-context LQR streams.
Stationary learning acceptance exists for the core feed-forward SAC/PPO paths.

Canonical task-aware Continual World and strict TA-CW10/TA-CW20 now have
runnable Meta-World adapters, training runners, evaluation matrices, configs,
RNG isolation, and simulator cleanup. The full one-million-step-per-occurrence
CW runs remain **not executed** in the repository evidence registry.

CARL has a strict hidden-context adapter and protocol configs, but a real CARL
smoke/benchmark run remains pending.

EWC, Online EWC, Synaptic Intelligence, and MAS have tested mathematical
regularizer mechanisms. Their actor/critic RL integration and declared
task-agnostic consolidation triggers remain pending, so they are not yet
complete RL baselines.
