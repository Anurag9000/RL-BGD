# Regularization Baselines

RL-BGD separates the mathematical regularizer from the event that triggers
consolidation. This distinction is required for honest task-agnostic
evaluation.

## EWC

The EWC mechanism stores an anchor parameter vector and a non-negative
diagonal importance estimate. Its penalty is one half times the configured
strength times the importance-weighted squared displacement from each stored
anchor. The repository provides a multi-consolidation EWC implementation and
an empirical-Fisher diagonal estimator from recomputed scalar loss closures.

If consolidation is triggered by the true task boundary, the resulting method
is an oracle/boundary-aware baseline and must be labelled as such.

## Online EWC

Online EWC retains one anchor and merges importance using

    F <- gamma * F_old + F_new

before moving the anchor to the current parameters. Gamma is constrained to
the interval from zero to one.

## Synaptic Intelligence

SI records the optimization-path contribution after every parameter update:

    omega_i <- omega_i - g_i * Delta theta_i

At consolidation it adds the non-negative path contribution divided by squared
total displacement plus damping to the persistent importance estimate. The
penalty then discourages movement away from the new anchor.

The implementation deliberately requires an explicit consolidation call. A
strict task-agnostic experiment cannot secretly invoke that call from hidden
task switches; it needs a declared boundary-free trigger such as a fixed
schedule or a learner-observable detector.

## MAS

MAS estimates parameter importance from the sensitivity of the squared L2 norm
of model outputs. The implemented estimator averages absolute gradients of
one-half the output norm squared. Importance accumulates across consolidation
events, and the quadratic penalty is anchored at the most recent consolidated
parameters.

As with EWC and SI, a true task-boundary consolidation is an oracle reference,
not a strict task-agnostic baseline.

## RL integration policy

Actor and critic regularizers must be reported separately because their
objectives and data distributions differ. Empirical Fisher in RL is not
silently called a likelihood Fisher when it is derived from a policy,
Bellman, or surrogate objective. The experiment configuration must state which
loss/output generated importance, which parameters were regularized, how often
importance was refreshed, and what information triggered consolidation.


## Task-agnostic SAC integration

RegularizedSACAgent wires EWC, Online EWC, SI, and MAS into the SAC actor,
critics, or both. Strict task-agnostic runs use a fixed optimizer-update
interval as the consolidation trigger. The trigger depends only on the
learner's own update count; environment task IDs, hidden context values, and
switch callbacks are not consumed.

For EWC/Online-EWC, actor importance uses squared gradients of the current SAC
policy surrogate and critic importance uses squared TD-loss gradients. This is
reported as an RL empirical-Fisher surrogate, not as an exact likelihood
Fisher. MAS uses actor-distribution and critic-output sensitivity. SI tracks
the unregularized task-loss gradients across optimizer steps before applying
its consolidation penalty.

The same agent exposes explicit consolidation for labelled oracle experiments,
but strict task-agnostic runners must not call it from true task boundaries.

Run the dependency-light hidden-context smoke path with:

    python scripts/run_regularized_sac_continual_lqr.py --method ewc
