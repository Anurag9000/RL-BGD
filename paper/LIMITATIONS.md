# Limitations and Scope

This document states limitations of the implemented RL-BGD research system
before final empirical conclusions are drawn.

## Bayesian approximation

RL-BGD uses a diagonal mean-field Gaussian posterior. Parameter correlations,
multimodal posteriors, structured matrix-variate uncertainty, and full natural
gradient geometry are not represented. Posterior standard deviation is
therefore an approximation whose functional meaning must be measured rather
than assumed.

BGD is computationally more expensive than a deterministic optimizer because
each Bayesian update uses Monte Carlo parameter samples and multiple
forward/backward evaluations. The compute/performance tradeoff is an explicit
paper metric.

## Generalized-Bayes RL interpretation

SAC and PPO losses are used as generalized-Bayes energies, not claimed to be
true data likelihoods. Evidence temperature is a modeling/calibration parameter.
The repository provides controlled sweeps but does not claim a unique
probabilistically correct temperature for actor-critic learning.

## Replay evidence

There is no assumed theorem saying that repeatedly sampling an old transition
from a replay buffer constitutes independent Bayesian evidence. The
fresh-only, inverse-reuse, and normalized evidence modes are experimental
corrections. Their usefulness must be established empirically and may depend on
buffer composition, off-policy distribution shift, replay ratio, and optimizer
details.

## Controlled forgetting

Posterior tempering intentionally gives up some accumulated posterior
information to recover plasticity. This can improve adaptation or damage
retention. Adaptive surprise is only a proxy for environmental change; reward
noise, policy improvement, bootstrapping error, or model misspecification can
all produce surprise without a task switch.

The predictive-NLL variant uses a learned diagonal-Gaussian transition/reward
model. It can be poorly calibrated, especially in high-dimensional observation
spaces, and is not a substitute for a full world-model CRL algorithm.

## Uncertainty interpretation

Small posterior sigma may correlate with curvature, importance, or resistance
to later movement, but those relationships are hypotheses. The mechanistic
suite tests them on controlled quadratics and causal freezing interventions.
Positive synthetic evidence does not automatically transfer to large neural
policies or critics.

## Task-agnostic scope

Strict task-agnostic runs remove task IDs, ground-truth context, and boundary
callbacks from learner inputs. This does not mean the environment is
stationary, nor that the learner has no temporal cues: observations, rewards,
previous actions, termination signals, replay statistics, and recurrent hidden
state can all contain inferential information about latent context.

Canonical Continual World and oracle-boundary UCL/EWC-style protocols use
privileged information by design and are labelled separately.

## Recurrence

Recurrent agents can infer hidden context from interaction history, but a finite
GRU state and truncated sequence training impose memory limits. Burn-in and
truncated BPTT approximate long histories. Recurrence also adds capacity and
optimization differences, so matched recurrent Adam controls are required
before attributing gains to Bayesian consolidation.

## Benchmark modernization

Continual World's historical software stack and some CORA dependencies are
obsolete relative to the modern base environment. RL-BGD uses modern
Meta-World adapters and documents deviations; it does not claim binary
reproduction of historical dependency stacks.

ContinualBench source distributions expose legacy asset-layout and runtime
defects. RL-BGD confines compatibility handling to the adapter boundary:
referenced missing assets are restored from canonical Meta-World paths, the
pinned debug-only undefined symbol is supplied without replacing reward
computation, and the upstream no-op close stub is tolerated. The pinned live
reset/step workflow now passes, but the integration remains dependent on
legacy Gym/MuJoCo internals and third-party benchmark behavior.

CORA metric/protocol compatibility is implemented, and the legacy **Atari**
runtime is validated in a separate Python 3.10 workflow pinned to CORA revision
`f2754bb282757829765beb4703f24b87efa13ff9`, NumPy 1.23.5, Gym 0.25.2,
ALE, and AutoROM. This is deliberately not a dependency of the modern RL-BGD
environment. The smoke does not validate CORA's other historical environment
families or reproduce published CORA results.

## External validity

Synthetic LQR/quadratic experiments are mechanism tests, not substitutes for
robotic or high-dimensional CRL results. CARL, CW10/CW20, and ContinualBench
are required to probe broader validity.

At the time of this document, the full one-million-step-per-occurrence
multi-seed Continual World confirmation runs have not been executed in the
repository evidence registry. No final comparative claim should be made from
their runnable configurations alone.

## Statistical limitations

Bootstrap intervals quantify variation across the executed seed set; they do
not correct for poor task coverage, biased hyperparameter selection, simulator
bugs, or protocol leakage. Expensive benchmark suites currently target fewer
seeds than cheap synthetic studies. Task scores within a seed are correlated,
which is why the paper builder uses hierarchical seed-then-task bootstrap when
appropriate.

Hyperparameters must be selected on development settings rather than repeatedly
optimized against the final CW20 aggregate. Changed protocols or failed seeds
must remain visible in provenance.

## Reproducibility limits

The project seeds Python, NumPy, PyTorch CPU/CUDA, environments, replay, and
task streams where supported. Bitwise reproduction is not guaranteed across GPU
architectures, CUDA/cuDNN versions, MuJoCo versions, or third-party benchmark
implementations.

The canonical artifact schema records source hashes and git revisions, but it
cannot by itself guarantee that external package registries or operating-system
drivers remain indefinitely available.

## Novelty and literature coverage

The literature review is a living audit. Failure to find a prior method is not
used as proof of novelty. Before submission, the final method combination and
claims must be checked again against newly published 2026 CRL/Bayesian-RL
literature and any implementation code reused from external projects must have
its license and provenance recorded explicitly.
