# RL-BGD

Research-grade infrastructure for studying **Bayesian Gradient Descent (BGD)** and generalized online Bayesian mechanisms in **task-agnostic continual reinforcement learning**.

The central question is whether posterior uncertainty can provide a task-boundary-free stability/plasticity mechanism for agents operating under nonstationary, recurring, and hidden-context MDPs, and where that mechanism fails due to posterior over-consolidation or replay evidence reuse.

## Status

The repository is being built in verified phases. The current foundation contains the first mathematically tested diagonal-Gaussian posterior, exact Gaussian tempering, Monte Carlo BGD update engine, diagnostics, device/seeding utilities, capability ledger, and research documentation. RL agents and benchmark adapters remain explicitly tracked as incomplete until wired and tested.

No experimental claims or benchmark numbers are fabricated. A capability is marked complete only when source, wiring, tests, and a runnable path exist.

## Core BGD update

For a factorized posterior q(theta) = product_i N(theta_i | mu_i, sigma_i^2), sample theta_i = mu_i + sigma_i epsilon_i, estimate g_bar = E[g] and c = E[g epsilon], then update

    mu_i <- mu_i - eta * sigma_i^2 * g_bar_i

    sigma_i <- sigma_i * sqrt(1 + (0.5 * sigma_i * c_i)^2)
               - 0.5 * sigma_i^2 * c_i

Posterior states are maintained in FP32 and clamped to configured uncertainty bounds.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/smoke_test.py
```

The smoke test runs a small quadratic BGD optimization and prints posterior diagnostics; it does not require MuJoCo or external benchmark downloads.

## Research design

The project separates Bayesian posterior mechanics, controlled tempering, replay evidence accounting, SAC/PPO backbones, recurrent hidden-context inference, information-access controls, benchmarks, evaluation, and paper artifacts.

See `docs/architecture.md`, `docs/mathematics.md`, `docs/literature_review.md`, and `docs/CAPABILITY_LEDGER.md`.

## License

MIT. Third-party algorithms and benchmarks retain their own licenses and are attributed separately.
