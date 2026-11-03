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

Unit coverage includes reset-mask causality, sequence-chunk hidden-state preservation, Adam/BGD recurrent PPO updates, recurrent SAC sequence replay with burn-in/unroll semantics, recurrent BGD-SAC actor/critic/all posterior modes, evidence accounting, and checkpoint round trips. Recurring hidden-context LQR runners verify that context changes do not create extra recurrent resets. Matched stationary learning-acceptance gates exist for recurrent Adam/BGD PPO and recurrent BGD-SAC. The 3RL-style Continual World path is runnable for recurrent Adam/BGD/adaptive-BGD; its full multi-million-step benchmark executions remain intentionally unexecuted until compute-backed raw artifacts exist.

## Memory budgeting for large recurrent replay

The 3RL-style Continual World configuration supports replay capacities up to
10 million transitions. CUDA-first training does **not** imply that all of
those transitions necessarily fit in available GPU VRAM. The deterministic
`rl_bgd.replay.estimate_sequence_replay_memory` function reports payload
bytes for observation/action/reward tensors, terminal and episode flags,
transition IDs, insertion provenance, evidence-use counters, full-capacity
checkpoint payload and an approximate per-update sampled batch allocation.

For FP32 replay, the persistent per-transition payload is exactly
`4 * (2 * observation_dim + action_dim + 1) + 3 * 8 + 4` bytes. This excludes
PyTorch allocator overhead, sampled indices and recurrent model activations,
host/GPU transfer staging, and external simulator storage. The checkpoint
figure is an upper bound on replay tensor payload for a full buffer, not an
on-disk compressed-file-size promise; peak restore RAM can substantially
exceed resident storage.

This diagnostic never silently shrinks capacity. The optional
`RecurrentSACTrainConfig.replay_storage` control provides `model` (default,
unchanged device/sampling behavior), `cpu` (CPU-resident replay even when
the actor and critic use CUDA), and `auto`. For a CUDA model, `auto` retains
replay on GPU only if estimated persistent replay storage, sampled batches
and sampling indices consume at most half of currently free VRAM; otherwise
the full configured capacity is kept on CPU. The model always trains on its
selected CUDA device, and **all** sampled tensor fields are moved to that
device with their original integer/boolean dtypes preserved.

A CPU replay with a CUDA model uses a separate CPU sampling generator;
the original action generator and original model-resident sampling route are
unchanged. Since CPU and CUDA random permutation implementations can differ,
the storage policy is part of the experiment configuration and may change
sample sequence under otherwise matched seeds. Checkpoint version 2 preserves
both RNG streams and the resolved storage-device identity, refuses mismatches,
and loads CPU-resident replay payloads without staging the complete buffer
through VRAM. Version 1 checkpoints remain loadable for default
`model`-resident replay. A changing `auto` device decision across a restart
is rejected rather than incorrectly certified as exact recovery.

These are new bounded software contracts, not a claim that a 10-million-entry
buffer has been physically allocated or that GPU-memory admission has been
validated under a full benchmark load.


For explicit CPU/GPU replay storage parity tests, `SequenceReplayBuffer`
accepts an index generator on either device and moves only the sampled
index vector to replay storage. Supplying two identically seeded CPU index
generators to CPU-resident and CUDA-resident replay yields identical logical
sample IDs and evidence-use counts; default model-device sampling is
otherwise unchanged. CUDA parity tests are conditional on real GPU access.
