# Bayesian Algorithms and Implementation Mapping

## Diagonal BGD

DiagonalGaussianPosterior stores one FP32 mean and standard deviation per trainable parameter plus the original prior. BGDUpdater draws reparameterized samples, evaluates gradients of the supplied objective, accumulates g_bar = E[g] and c = E[g * epsilon], and applies the diagonal BGD mean/std update with explicit sigma bounds.

Antithetic sampling is optional and requires an even Monte Carlo count. K=1,2,4,8 are supported; K=1 is used with antithetic sampling disabled.

## Posterior tempering

Before a BGD evidence update, configured retention lambda < 1 can replace the current diagonal Gaussian with the exact normalized product q_prev(theta)^lambda * p0(theta)^(1-lambda). The implementation works in precision space and is unit-tested.

## BGD-SAC

BGDSACAgent supports critic_only, actor_only, and actor_and_critic.

The Bellman target is computed first using the policy at its current mean parameters and deterministic target critic mean networks. For a Bayesian critic, each sampled critic parameter set is trained against that fixed target through the BGD objective. After the posterior update, the live critic module is synchronized to the posterior mean.

For a Bayesian actor, sampled actor parameters are evaluated through torch.func.functional_call; the SAC policy objective alpha * log pi(a|s) - min(Q1(s,a), Q2(s,a)) is differentiated with respect to the sampled actor parameters. Critics are fixed functions for that actor update. Entropy temperature alpha continues to use its standard scalar Adam update.

Target critics are Polyak-updated from posterior means only. Full posterior targets remain an explicit future ablation instead of being silently enabled.

## Generalized-Bayes interpretation

The repository does not claim SAC losses are literal supervised likelihoods. BGD-SAC is treated as a generalized-Bayesian update using the SAC surrogate as an energy/loss. Replay evidence reuse and posterior tempering are separate research dimensions.
