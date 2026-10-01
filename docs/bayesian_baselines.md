# Bayesian baseline mapping: FOO-VB and UCL

## FOO-VB Diagonal

Zeno et al. derive fixed-point equations for online variational Bayes. In the
diagonal Gaussian case, the one-step fixed-point update for the posterior mean
and standard deviation is algebraically the same update already implemented by
the repository's diagonal BGD engine when:

- mean-step eta is exactly 1;
- posterior tempering is disabled;
- the current posterior is carried forward online.

For that reason, RL-BGD does not maintain a second copy of identical numerical
code. The named FOO-VB diagonal baseline is created through
make_foo_vb_diagonal_updater and is protected by an exact equality test against
BGD configured with eta=1.

The original FOO-VB paper derives the update for an online Bayesian likelihood
objective. Applying the same operator to SAC or PPO losses is therefore labelled
a generalized-Bayes FOO-VB-style RL baseline, not an exact reproduction of the
paper's supervised likelihood experiments.

The official full FOO-VB repository also implements matrix-variate Gaussian
updates with layerwise A/B covariance factors. That structured posterior is a
separate future baseline and is not silently conflated with the diagonal
variant.

## UCL

The original UCL implementation uses saved previous-task Bayesian parameters,
explicit task numbers, and task-specific RL heads. It is therefore an
oracle/task-boundary baseline under this repository's information-access
taxonomy.

Any future UCL integration must preserve that label unless a distinct
task-agnostic adaptation is implemented and named as an adaptation rather than
the original algorithm.

Primary implementations consulted:

- https://github.com/chenzeno/FOO-VB
- https://github.com/csm9493/UCL

## UCL-PPO oracle-boundary baseline

The original UCL reinforcement-learning code is explicitly task/boundary aware:
the training driver iterates named tasks, passes task_num into the policy, and
after every task calls update_old_actor_critic(). The saved previous-task
posterior is then used by PPO-UCL's uncertainty-guided regularizer.

RL-BGD therefore labels UCL-PPO as an oracle-boundary comparator. It does not
place it in the strict task-agnostic column.

The implementation preserves the source mechanism:

- Gaussian Bayesian hidden layers;
- Adam PPO optimization, not the BGD fixed-point optimizer;
- a frozen previous-task posterior snapshot;
- mean-change penalties scaled by saved uncertainty and previous-layer
  uncertainty;
- the saved-posterior L1 term after the first completed task;
- variance-ratio and variance-normalization penalties controlled by beta;
- deterministic action/value output heads, matching the original design where
  UCL regularization excludes the policy distribution and critic output heads.

The official RL source contains an internal mismatch: ppo_ucl.py references
bias_mu and bias_rho while its checked-in BayesianLinear exposes a deterministic
bias only. RL-BGD repairs that inconsistency by giving bias an explicit
Gaussian posterior and testing its regularization/checkpoint behavior.

Runnable command:

    python scripts/run_ucl_ppo_lqr.py

This baseline receives true phase boundaries solely to snapshot the previous
posterior. It receives no task ID and uses no task-specific head in the RL-BGD
synthetic runner, so it is stricter than the original task-indexed UCL policy,
but it is still not task-agnostic because boundary access remains oracle.
