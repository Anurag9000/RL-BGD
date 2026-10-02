# Mechanistic analysis suite

Phase 13 is implemented as one deterministic synthetic analysis pipeline:

    python scripts/run_mechanistic_analysis.py --output-dir artifacts/mechanistic

The runner creates raw CSV data, summary JSON, and figures without manual
transcription.

## Analyses

1. Movement versus sigma: consolidate an anisotropic diagonal quadratic, record
posterior sigma, apply an abrupt optimum shift, and measure absolute parameter
movement after the first BGD update.

2. Perturbation importance: perturb every parameter by the same fixed amount at
the consolidated solution and compare the induced loss increase with posterior
precision.

3. Freezing counterfactual: freeze either the lowest-sigma or highest-sigma
quartile during matched post-shift adaptation and compare against a no-freeze
control under the same Monte-Carlo seed.

4. Curvature signal: estimate E[g epsilon] on a known diagonal quadratic and
compare it dimension-wise with the analytic small-sigma target H sigma.

5. Uncertainty quality: report rank correlations and sigma-quartile diagnostics
linking uncertainty to future movement and perturbation sensitivity.

6. Late-life plasticity stress: repeatedly consolidate BGD on a fixed quadratic,
then apply a controlled abrupt optimum shift while holding the post-shift update
rule at retention 1.0. The matched `ablation_core` arms compare vanilla
pre-shift retention 1.0 against controlled pre-shift retention 0.97 over the
same five seeds and all other settings. The runner records pre-shift sigma and
effective learning rate, first-step movement, post-shift normalized AUC,
recovery fraction, final target loss, and the full step-indexed normalized loss
timeline. The task shift is applied by the experimental harness and is recorded
explicitly in information-access metadata; it is not presented as a
boundary-free online change detector.

## Outputs

- mechanistic_parameters.csv
- freezing_counterfactual.csv
- mechanistic_summary.json
- movement_vs_sigma.png
- perturbation_vs_precision.png
- freezing_counterfactual.png
- curvature_signal.png
- uncertainty_quality.png

The Phase-13 command above produces the parameter-level files/figures directly.
Late-plasticity matched arms are executed through the curated `ablation_core`
suite so each seed is written to the canonical run schema and the normalized
adaptation timeline can be aggregated with bootstrap bands by the Phase-15 paper
builder.

These are controlled synthetic mechanistic diagnostics. They do not by
themselves establish the same relationships in SAC/PPO neural networks.
