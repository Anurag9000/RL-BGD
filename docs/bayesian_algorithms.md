# Bayesian Algorithms and Implementation Mapping

## Diagonal BGD

DiagonalGaussianPosterior stores one FP32 mean and standard deviation per trainable parameter plus the original prior. BGDUpdater draws reparameterized samples and maintains two explicit gradient channels: the ordinary objective gradient controls posterior-mean learning, while the uncertainty/evidence gradient controls c = E[g_evidence * epsilon] and therefore sigma. With the default scalar objective both channels are identical and recover vanilla BGD.

Antithetic sampling is optional and requires an even Monte Carlo count. K=1,2,4,8 are supported; K=1 is used with antithetic sampling disabled.

## Posterior tempering

Before a BGD evidence update, configured retention lambda < 1 can replace the current diagonal Gaussian with the exact normalized product q_prev(theta)^lambda * p0(theta)^(1-lambda). The implementation works in precision space and is unit-tested.

## Replay evidence accounting

Off-policy replay can expose the same transition to a Bayesian updater many times. RL-BGD keeps the ordinary replay objective for posterior means while independently controlling how much each use contributes to uncertainty consolidation.

Implemented modes:

- all_replay: every sampled use has evidence weight 1.
- fresh_only_uncertainty: only the first sampled use of a transition affects sigma; replay still trains mu.
- inverse_reuse_weight: the uncertainty loss for transition j is weighted by 1 / usage_count_j.
- normalized_batch_evidence: within-batch optimization composition is unchanged, but the whole uncertainty update is scaled by the batch mean of 1 / usage_count.

The weighted uncertainty loss is mean(weight_j * loss_j), not a normalized weighted mean. This distinction is deliberate: a smaller weight represents less accumulated Bayesian evidence rather than redistributing a fixed evidence budget. Effective sample size, freshness, usage counts, and evidence-weight statistics are logged.

These modes are evidence-accounting hypotheses rather than claims that one exactly recovers a unique "correct" posterior under replay.

## BGD-SAC

BGDSACAgent supports critic_only, actor_only, and actor_and_critic.

The Bellman target is computed first using the policy at its current mean parameters and deterministic target critic mean networks. For a Bayesian critic, each sampled critic parameter set is trained against that fixed target through the BGD objective. After the posterior update, the live critic module is synchronized to the posterior mean.

For a Bayesian actor, sampled actor parameters are evaluated through torch.func.functional_call; the SAC policy objective alpha * log pi(a|s) - min(Q1(s,a), Q2(s,a)) is differentiated with respect to the sampled actor parameters. Critics are fixed functions for that actor update. The same sampled action is used to construct both the mean-learning and replay-evidence losses. Entropy temperature alpha continues to use its standard scalar Adam update.

Target critics are Polyak-updated from posterior means only. Full posterior targets remain an explicit future ablation instead of being silently enabled.

## Generalized-Bayes interpretation

The repository does not claim SAC losses are literal supervised likelihoods. BGD-SAC is treated as a generalized-Bayesian update using the SAC surrogate as an energy/loss. Replay evidence reuse and posterior tempering are separate research dimensions.
