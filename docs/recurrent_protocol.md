# Recurrent Hidden-Context Protocol

Recurrent PPO is implemented as a matched architecture family for Adam and Bayesian Gradient Descent. Both variants use the same observation encoder, single GRUCell state dimension, action/value heads, rollout segmentation, and episode reset semantics. Optimizer/posterior mechanics are the intended experimental difference.

## Hidden-state semantics

The hidden state is reset only at natural environment episode boundaries (terminated or truncated). A hidden task/context switch in a strict continual stream does not reset the recurrent state because the learner is not told that a switch occurred.

Each rollout stores the actor and value hidden state immediately before every observation together with an episode_start mask. PPO updates use contiguous sequence chunks and truncated backpropagation through time. If a chunk begins mid-episode, its initial hidden state is the behavior-policy state captured during collection. Episode-start masks inside the chunk reset state to zero.

After an update, the just-collected rollout is replayed through the updated networks to refresh the live hidden state before the next rollout. This reduces cross-rollout stale-state drift. The first hidden state of a rollout remains the standard truncated-BPTT approximation inherited from prior history.

## PPO behavior-policy correctness

Old action log-probabilities and old values are frozen at collection time. Recurrent BGD does not resample a new behavior policy during collection. Bayesian parameter samples are used only inside PPO update objectives.

Repeated PPO epochs are also repeated evidence. BGD recurrent PPO therefore shares the explicit evidence modes used by feed-forward BGD-PPO: first_epoch_only (default), all_epochs, and normalized_epochs.

## Information access

Strict recurrent continual runs receive observations, actions, rewards, termination/truncation signals, and their own hidden state only. They do not receive task IDs, task names, boundary callbacks, context values, or task-routed resets. Evaluator-owned context remains separate.

## Current validation status

Unit coverage includes reset-mask causality, sequence-chunk hidden-state preservation, Adam recurrent PPO updates, BGD recurrent posterior updates, and checkpoint round trips. The recurring LQR runner verifies that context changes do not create extra recurrent resets. Longer learning-acceptance experiments and recurrent SAC/sequence replay remain follow-up work.
