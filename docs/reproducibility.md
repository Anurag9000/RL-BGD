# Reproducibility

Future training runs must store resolved configuration, config hash, git revision/dirty state, Python/PyTorch/CUDA/device metadata, dependency versions, seeds, environment seed, task order, training budget, and raw metrics.

Checkpoint versioning starts at version 1 for Bayesian posterior/updater state. Full RL checkpoints must later add model/targets/optimizers/replay when requested/RNG/stream/recurrent/surprise/normalization states.

Bitwise equality across GPU architectures is not promised.
