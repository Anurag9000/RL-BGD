# RL-BGD

Research-grade infrastructure for studying **Bayesian Gradient Descent (BGD)** and generalized online Bayesian mechanisms in **task-agnostic continual reinforcement learning**.

The central question is whether posterior uncertainty can provide a task-boundary-free stability/plasticity mechanism for agents operating under nonstationary, recurring, and hidden-context MDPs, and where that mechanism fails due to posterior over-consolidation or replay evidence reuse.

## Status

The repository now contains tested SAC/PPO and BGD variants, recurrent hidden-context agents, replay-evidence controls, fixed/adaptive posterior tempering, CARL and Continual World integrations, external continual-learning baselines, mechanistic analysis, curated paper suites, and an automatic raw-run-to-paper artifact pipeline.

Large CW10/CW20 confirmation runs and any other expensive studies remain explicitly **unexecuted** until raw artifacts exist. Optional legacy benchmark runtimes are isolated when their dependency stacks conflict with the modern base environment.

No experimental claims or benchmark numbers are fabricated. A capability is marked complete only when source, wiring, tests, and a runnable path exist; execution and scientific support are tracked separately.

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
python scripts/run_paper_suite.py smoke --output-root artifacts/suites
python scripts/build_paper_artifacts.py --results-root artifacts/suites --output-dir artifacts/paper
```

The smoke test runs a small quadratic BGD optimization and prints posterior diagnostics; it does not require MuJoCo or external benchmark downloads.

## Research design

The project separates Bayesian posterior mechanics, controlled tempering, replay evidence accounting, SAC/PPO backbones, recurrent hidden-context inference, information-access controls, benchmarks, evaluation, and paper artifacts.

See `docs/architecture.md`, `docs/mathematics.md`, `docs/literature_review.md`, and `docs/CAPABILITY_LEDGER.md`.

## License

MIT. Third-party algorithms and benchmarks retain their own licenses and are attributed separately.
