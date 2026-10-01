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
