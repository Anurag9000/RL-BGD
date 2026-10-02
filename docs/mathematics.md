# Mathematics

## Diagonal BGD

For q(theta)=product_i N(theta_i|mu_i,sigma_i^2), sample theta_i=mu_i+sigma_i epsilon_i. With gradients g_i^(k),

    g_bar_i = mean_k g_i^(k)
    c_i     = mean_k g_i^(k) epsilon_i^(k)

The implemented updates are

    mu_i <- mu_i - eta sigma_i^2 g_bar_i

    sigma_i <- sigma_i sqrt(1 + (0.5 sigma_i c_i)^2)
               - 0.5 sigma_i^2 c_i

For a locally quadratic objective and sufficiently small sigma, E[g_i epsilon_i] approximates H_ii sigma_i. Mathematical tests verify the uncertainty-change sign under positive/negative quadratic curvature and numerically check this relation.

## Controlled posterior tempering

With q_prev=N(mu,sigma^2), p0=N(mu0,sigma0^2), and retention lambda,

    q_tilde proportional to q_prev^lambda p0^(1-lambda)

so in precision space

    tau_tilde = lambda tau + (1-lambda) tau0

    mu_tilde = [lambda tau mu + (1-lambda) tau0 mu0] / tau_tilde

    sigma_tilde = tau_tilde^(-1/2)

The implementation supports lambda in [0,1]. Fixed/adaptive schedules call this primitive rather than duplicating the formula.

## Generalized Bayesian RL

RL surrogate objectives are not silently presented as literal negative log likelihoods. The implemented interpretation is the generalized posterior objective

    q_u = argmin_q E_q[L_RL,u(theta)] + (1/beta_u) KL(q || q_prior,u)

with positive evidence temperature beta. In BGDUpdater, beta multiplies the mean-learning and uncertainty-evidence objectives before their gradients are formed. The reported unscaled RL loss is unchanged. Consequently, beta changes both the posterior-mean gradient and the curvature signal that controls sigma; it is not generally equivalent to changing eta. A matched quadratic mechanism test and SAC temperature sweep expose this distinction.

SAC computes a fixed Bellman target from mean policy/target-critic parameters before sampled critic updates. Bayesian actor samples use the ordinary SAC policy surrogate against fixed current critics. Target critics track posterior means through Polyak averaging. PPO freezes behavior log probabilities/values at collection time and treats repeated rollout epochs as an explicit evidence-reuse dimension.

Replay-evidence accounting and posterior tempering are separate from beta: replay weights can change only the uncertainty/evidence loss, while exact Gaussian tempering modifies the prior/posterior state before the next BGD evidence update.
