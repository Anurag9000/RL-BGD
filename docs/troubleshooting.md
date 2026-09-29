# Troubleshooting

- Requesting CUDA on a CPU-only runtime fails loudly; use auto or cpu.
- Antithetic BGD sampling requires an even Monte Carlo sample count.
- Invalid sigma bounds or non-finite posterior state raise errors.
- Optional benchmark import failures should be resolved via the relevant extra; base math tests must not import MuJoCo/CARL/Meta-World.
