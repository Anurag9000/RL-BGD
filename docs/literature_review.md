# Literature Review — Living Research Authority

This document records the primary sources that justify RL-BGD's implemented
methods, baselines, benchmarks, and mechanistic hypotheses. It separates
published source facts from RL-BGD design choices. Absence from this audit is
never treated as evidence of novelty.

## 1. Bayesian continual-learning foundations

### Bayesian Gradient Descent (BGD)

Chen Zeno, Itay Golan, Elad Hoffer, and Daniel Soudry, *Task Agnostic
Continual Learning Using Online Variational Bayes* (2018).

BGD is RL-BGD's direct mathematical starting point: a mean-field Gaussian
posterior is updated online without requiring known task boundaries. The paper
does not establish the semantics of off-policy replay as Bayesian evidence,
hidden-context inference, SAC/PPO surrogate likelihoods, or controlled
posterior forgetting in continual RL; those must therefore be tested
separately here.

Source: https://arxiv.org/abs/1803.10123

### FOO-VB fixed-point online variational Bayes

Chen Zeno, Itay Golan, Elad Hoffer, and Daniel Soudry, *Task Agnostic
Continual Learning Using Online Variational Bayes with Fixed-Point Updates*
(2020).

FOO-VB derives fixed-point equations for the online variational-Bayes
optimization problem for multivariate Gaussian parameter distributions. It is
the main authority for RL-BGD's exact diagonal fixed-point/equivalence
reference tests and for interpreting BGD as an approximation to an online
posterior update rather than as an ordinary optimizer with noise.

Source: https://arxiv.org/abs/2010.00373

### Variational Continual Learning (VCL)

Cuong V. Nguyen, Yingzhen Li, Thang D. Bui, and Richard E. Turner,
*Variational Continual Learning*, ICLR 2018.

VCL formalizes the recurring Bayesian idea that the approximate posterior after
one task becomes the prior for subsequent learning, with practical coreset
variants. It is important background for posterior propagation, but its normal
task-sequential formulation does not by itself supply RL-BGD's strict
task-agnostic protocol.

Source: https://arxiv.org/abs/1710.10628

### Generalized Bayesian updating

P. G. Bissiri, C. C. Holmes, and S. G. Walker, *A General Framework for
Updating Belief Distributions*, JRSS-B 2016.

This work provides the decision-theoretic authority for posterior-like updates
based on a loss rather than a literal log likelihood, including calibration of
the relative weight of loss information against prior information. RL-BGD uses
this as the conceptual basis for treating SAC/PPO objectives as generalized
Bayes losses and for explicitly sweeping an evidence temperature. It does not
imply that any particular RL surrogate or temperature is uniquely correct.

Source: https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12158

## 2. Parameter-importance and distillation baselines

### Elastic Weight Consolidation (EWC)

James Kirkpatrick et al., *Overcoming Catastrophic Forgetting in Neural
Networks*, PNAS 2017.

EWC protects parameters estimated to be important for previous behavior using
a Fisher-weighted quadratic penalty. The original work includes reinforcement
learning experiments. RL-BGD's EWC comparator is therefore a relevant
parameter-consolidation baseline, but oracle boundary-triggered EWC is labelled
as privileged whenever true task boundaries are supplied.

Source: https://www.pnas.org/doi/10.1073/pnas.1611835114

### Online EWC / Progress & Compress

Jonathan Schwarz et al., *Progress & Compress: A Scalable Framework for
Continual Learning*, ICML 2018.

Progress & Compress uses an active column and a persistent knowledge base, with
online consolidation that keeps model size constant. It is the principal
authority for the online-EWC style importance accumulation used as a baseline
here. RL-BGD also tests fixed-update task-agnostic consolidation separately
from boundary-triggered variants so that boundary information is not silently
introduced.

Source: https://proceedings.mlr.press/v80/schwarz18a.html

### Synaptic Intelligence (SI)

Friedemann Zenke, Ben Poole, and Surya Ganguli, *Continual Learning Through
Synaptic Intelligence*, ICML 2017.

SI accumulates parameter importance from the optimization path and penalizes
movement of parameters that contributed strongly to prior learning. RL-BGD's
SI implementation follows this path-integral/consolidation distinction and
keeps the consolidation trigger explicit.

Source: https://proceedings.mlr.press/v70/zenke17a.html

### Memory Aware Synapses (MAS)

Rahaf Aljundi et al., *Memory Aware Synapses: Learning What (not) to Forget*,
ECCV 2018.

MAS estimates parameter importance from the sensitivity of the learned
function to parameter changes, allowing importance estimation without labels.
It motivates RL-BGD's output-sensitivity MAS baseline and the broader
mechanistic question of whether posterior uncertainty tracks functional
parameter importance.

Source:
https://openaccess.thecvf.com/content_ECCV_2018/html/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.html

### Learning without Forgetting (LwF)

Zhizhong Li and Derek Hoiem, *Learning without Forgetting* (2016).

LwF preserves prior capabilities through distillation on new-task inputs when
old training data are unavailable. It is useful distillation background and a
possible future comparator, but RL-BGD does not currently present LwF as an
implemented task-agnostic RL baseline.

Source: https://arxiv.org/abs/1606.09282

## 3. Bayesian uncertainty, consolidation, and controlled forgetting

### Uncertainty-based Continual Learning (UCL)

Hongjoon Ahn, Sungmin Cha, Donggyu Lee, and Taesup Moon,
*Uncertainty-based Continual Learning with Adaptive Regularization* (2019).

UCL uses Bayesian parameter uncertainty to regulate plasticity and reports
lifelong reinforcement-learning experiments. RL-BGD includes a separately
labelled oracle-boundary PPO-UCL baseline. UCL is especially relevant to the
question of whether low uncertainty should imply stronger consolidation.

Source: https://arxiv.org/abs/1905.11614

### AFEC active forgetting

Liyuan Wang et al., *AFEC: Active Forgetting of Negative Transfer in Continual
Learning*, NeurIPS 2021.

AFEC explicitly argues that preserving old knowledge can cause negative
forward transfer and introduces a Bayesian active-forgetting construction with
synaptic expansion/convergence. Its experiments include Atari reinforcement
tasks. It supports the broader scientific motivation for testing controlled
forgetting rather than assuming maximal posterior retention is always optimal.

Source:
https://proceedings.neurips.cc/paper/2021/hash/bc6dc48b743dc5d013b1abaebd2faed2-Abstract.html

### MESU and catastrophic remembering

Djohan Bonnet et al., *Bayesian Continual Learning and Forgetting in Neural
Networks*, Nature Communications 2025.

The paper introduces Metaplasticity from Synaptic Uncertainty (MESU), explicitly
targets the tension between catastrophic forgetting and catastrophic
remembering, and develops controlled forgetting from a Bayesian continual
learning perspective without requiring explicit task boundaries. This is a
direct modern authority for RL-BGD's long-horizon uncertainty-collapse and
replasticization questions. RL-BGD does not claim its Gaussian tempering rule
is MESU.

Source: https://www.nature.com/articles/s41467-025-64601-w

## 4. Plasticity loss as a distinct failure mode

### Continual deep-RL plasticity loss

Zaheer Abbas et al., *Loss of Plasticity in Continual Deep Reinforcement
Learning*, CoLLAs 2023.

The paper demonstrates diminishing ability to learn as deep RL agents cycle
through Atari tasks and studies changes in gradients, weights, and activation
sparsity. This supports measuring late-life adaptation ability separately from
ordinary forgetting.

Source: https://proceedings.mlr.press/v232/abbas23a.html

### General deep continual-learning plasticity loss

Shibhansh Dohare et al., *Loss of Plasticity in Deep Continual Learning*,
Nature 2024.

The paper systematically distinguishes loss of plasticity from catastrophic
forgetting and demonstrates the phenomenon across supervised and reinforcement
learning settings. It motivates RL-BGD's explicit late-stream adaptation,
T80/T90, plasticity-retention, sigma-collapse, and effective-learning-rate
analyses.

Source: https://www.nature.com/articles/s41586-024-07711-7

## 5. Continual-reinforcement-learning formulations and benchmarks

### Continual-RL definitions and taxonomy

Khimya Khetarpal, Matthew Riemer, Irina Rish, and Doina Precup,
*Towards Continual Reinforcement Learning: A Review and Perspectives* (2020),
and David Abel et al., *A Definition of Continual Reinforcement Learning*
(2023), provide complementary taxonomy and formalization of non-stationary,
lifelong RL. They reinforce the distinction between merely solving a sequence
of labelled tasks and agents that must continue adapting indefinitely.

Sources:
- https://arxiv.org/abs/2012.13490
- https://arxiv.org/abs/2307.11046

### Policy Consolidation

Christos Kaplanis, Murray Shanahan, and Claudia Clopath, *Policy Consolidation
for Continual Reinforcement Learning*, ICML 2019.

Policy Consolidation uses a multi-timescale cascade of policies to stabilize
learning without relying on task-boundary callbacks. It is an important
boundary-free CRL comparator family.

Source: https://proceedings.mlr.press/v97/kaplanis19a.html

### CLEAR

David Rolnick et al., *Experience Replay for Continual Learning*, NeurIPS 2019.

CLEAR combines replay, off-policy learning, and behavioral cloning with
on-policy learning and does not require knowledge of individual task
boundaries. It is central prior work for replay-based CRL. It supports replay
as a retention mechanism; it does **not** establish that repeated replay uses
should be counted as independent Bayesian observations.

Source:
https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html

### Continual World

Maciej Wołczyk et al., *Continual World: A Robotic Benchmark for Continual
Reinforcement Learning*, NeurIPS 2021.

Continual World is a Meta-World-derived robotic benchmark designed to evaluate
both forgetting and forward transfer. RL-BGD maintains a canonical task-aware
protocol for comparability and a separately labelled strict task-agnostic
variant. The historical public implementation uses an older software stack, so
the repository's modern Meta-World adapter is compatibility engineering rather
than a claim of byte-for-byte reproduction.

Paper:
https://proceedings.neurips.cc/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html

Reference implementation: https://github.com/awarelab/continual_world

### Disentangling transfer / ClonEx-SAC

Maciej Wołczyk et al., *Disentangling Transfer in Continual Reinforcement
Learning*, NeurIPS 2022.

This work studies actor, critic, exploration, and replay-data contributions to
transfer under SAC on Continual World. It is important authority for RL-BGD's
actor-only/critic-only/actor-and-critic Bayesianization ablations and for
treating transfer as a first-class metric rather than evaluating forgetting
alone.

Source:
https://proceedings.neurips.cc/paper_files/paper/2022/hash/2938ad0434a6506b125d8adaff084a4a-Abstract-Conference.html

### 3RL task-agnostic continual RL

Massimo Caccia, Jonas Mueller, Taesup Kim, Laurent Charlin, and Rasool Fakoor,
*Task-Agnostic Continual Reinforcement Learning* (2022).

3RL combines replay with recurrent context inference for task-agnostic agents
and evaluates on synthetic settings and Meta-World. It is the key authority
for separating hidden-context inference from parameter consolidation. RL-BGD's
recurrent CW runner is explicitly labelled "3RL-style" and documents its
architectural deviations rather than claiming reproduction identity.

Source: https://arxiv.org/abs/2205.14495

### CARL

Carolin Benjamins et al., *CARL: A Benchmark for Contextual and Adaptive
Reinforcement Learning* (2021).

CARL exposes controlled changes in environment context across classic control,
physical simulation, games, and other domains. RL-BGD uses strict
hidden-context CARL streams to test adaptation to abrupt, smooth, and recurring
dynamics changes while keeping ground-truth context out of the agent's input.

Source: https://arxiv.org/abs/2110.02102

### CORA

Sam Powers et al., *CORA: Benchmarks, Baselines, and Metrics as a Platform for
Continual Reinforcement Learning Agents*, CoLLAs 2022.

CORA provides benchmark families and continual-evaluation metrics including
isolated forgetting and zero-shot forward transfer. RL-BGD provides a metric
compatibility bridge while intentionally isolating CORA's legacy runtime
dependencies from the modern base environment.

Paper: https://arxiv.org/abs/2110.10067

Reference code: https://github.com/AGI-Labs/continual_rl

## 6. World models for continual RL

### Continual-Dreamer

Samuel Kessler et al., *The Effectiveness of World Models for Continual
Reinforcement Learning*, CoLLAs 2023.

This work studies world models and selective replay for continual RL and
introduces Continual-Dreamer as a task-agnostic approach. It demonstrates that
world-model learning and replay constitute a distinct CRL design axis from
parameter-importance regularization.

Source: https://proceedings.mlr.press/v232/kessler23a.html

### Online world-model planning / ContinualBench

Zichen Liu, Guoji Fu, Chao Du, Wee Sun Lee, and Min Lin, *Continual
Reinforcement Learning by Planning with Online World Models*, ICML 2025.

The paper proposes an online shallow world model with model-predictive control
and introduces ContinualBench. RL-BGD uses ContinualBench as an external
strict-hidden-task integration target and separately implements learned
predictive-NLL surprise; this does not make RL-BGD's predictive surprise model
equivalent to the paper's planning agent.

Paper: https://proceedings.mlr.press/v267/liu25p.html

Reference environment: https://github.com/sail-sg/ContinualBench

## 7. Bayesian/randomized exploration context

### Posterior Sampling for Reinforcement Learning

Ian Osband, Daniel Russo, and Benjamin Van Roy, *(More) Efficient Reinforcement
Learning via Posterior Sampling* (2013).

PSRL samples an MDP from the posterior and acts according to its optimal policy
for an episode. It is the canonical Bayesian-exploration reference for why
temporally coherent posterior samples can drive exploration.

Source: https://arxiv.org/abs/1306.0940

### Bootstrapped DQN

Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy,
*Deep Exploration via Bootstrapped DQN* (2016).

Bootstrapped DQN demonstrates temporally extended exploration through
randomized value functions. It is relevant background for RL-BGD posterior
policy/value sampling experiments. Sampling a BGD weight posterior is **not**
claimed to be theoretically equivalent to exact PSRL or Bootstrapped DQN.

Source: https://arxiv.org/abs/1602.04621

## 8. Replay evidence semantics: established facts versus RL-BGD hypotheses

Published CRL work such as CLEAR, 3RL, and Continual-Dreamer establishes replay
as a powerful mechanism for retaining and transferring experience. Bayesian
continual-learning work establishes sequential posterior updating. These two
facts do not automatically imply that drawing the same transition from an
off-policy replay buffer many times should multiply its Bayesian evidence.

Accordingly, RL-BGD treats its replay-evidence modes as controlled modeling
choices to be compared empirically:

- all replay contributes to posterior mean and uncertainty updates;
- fresh-only uncertainty evidence;
- inverse-reuse-weighted uncertainty evidence;
- normalized-batch evidence.

The repository must not describe any of these corrections as a published
Bayesian theorem unless a direct source is added. Their purpose is to expose
and test the causal hypothesis that replay reuse can create posterior
overconfidence and reduce later plasticity.

## 9. Source/code-use policy

Paper equations and algorithmic ideas are research authority, not permission to
copy source code. RL-BGD's implementations are independent unless a repository
file explicitly records reused code and its license/provenance. External
benchmark packages (for example Meta-World, CARL, and ContinualBench) remain
optional dependencies and are integrated through adapters rather than vendored
source trees.

Before adding any future comparator by copying implementation code, record its
upstream repository, pinned revision, license, and exactly which source was
reused.

## Residual living-audit items

The core literature authority required by the currently implemented
experiments is covered above. This remains a living review, not a claim that
the CRL literature is exhausted. Future work should continue to check:

- newly published 2026 CRL methods before selecting final external
  comparators;
- direct work connecting generalized-Bayes loss temperatures specifically to
  modern actor-critic posterior updates;
- direct theory for Bayesian evidence accounting under repeated off-policy
  replay, if such work emerges;
- license/provenance review whenever implementation code, rather than only
  published ideas or package APIs, is reused.

Any later source that materially changes an implemented protocol or comparison
must be reflected in the capability ledger, experiment registry, and paper
method/limitations documents.
