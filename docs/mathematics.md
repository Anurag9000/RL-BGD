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

RL surrogate objectives are not silently presented as literal negative log likelihoods. The intended generalized posterior objective is

    q_u = argmin_q E_q[L_RL,u(theta)] + (1/beta_u) KL(q || q_prior,u)

SAC/PPO-specific sampling, target-network, replay-evidence, and tempering semantics must be stated when those phases are implemented.
