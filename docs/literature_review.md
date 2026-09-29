# Literature Review — Living Audit

This living document distinguishes verified source facts from planned reimplementations and does not claim novelty merely because an idea is absent from this initial audit.

## Task Agnostic Continual Learning Using Online Variational Bayes (BGD)
Chen Zeno, Itay Golan, Elad Hoffer, Daniel Soudry (2018). Mean-field Gaussian online variational learning targeting continual learning with unknown task boundaries. Original code is linked by the paper at github.com/igolan/bgd. This is RL-BGD's mathematical foundation, but does not by itself settle RL replay semantics, hidden-context inference, or generalized-Bayes policy/value objectives.
Source: https://arxiv.org/abs/1803.10123

## Variational Continual Learning
Cuong V. Nguyen, Yingzhen Li, Thang D. Bui, Richard E. Turner, ICLR 2018. Sequential approximate Bayesian learning using the previous variational posterior as the next prior; coreset variants are part of the practical treatment.
Source: https://arxiv.org/abs/1710.10628

## Uncertainty-based Continual Learning with Adaptive Regularization (UCL)
Hongjoon Ahn, Sungmin Cha, Donggyu Lee, Taesup Moon (2019). Variational Bayesian framing with node-wise uncertainty and plasticity/stability regularization; the paper reports lifelong RL experiments.
Source: https://arxiv.org/abs/1905.11614

## Policy Consolidation for Continual Reinforcement Learning
Christos Kaplanis, Murray Shanahan, Claudia Clopath, ICML 2019. A multi-timescale cascade of policy networks designed not to require task boundaries.
Source: https://proceedings.mlr.press/v97/kaplanis19a.html

## CLEAR / Experience Replay for Continual Learning
David Rolnick et al., NeurIPS 2019. Combines off-policy replay and behavioral cloning with on-policy learning, without requiring knowledge of individual tasks according to the paper abstract. It is a key replay-centric CRL comparator and motivates explicit Bayesian evidence-reuse accounting.
Source: https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html

## Continual World
Meta-World-derived continual robotic manipulation benchmark. The canonical public repository documents CW20 as 20 tasks with 1M steps per task. Its historical implementation depends on older TensorFlow/mujoco-py/Meta-World versions, so RL-BGD must verify modern compatibility rather than copy pins blindly.
Source: https://github.com/awarelab/continual_world

## 3RL — Task-Agnostic Continual Reinforcement Learning
Massimo Caccia, Jonas Mueller, Taesup Kim, Laurent Charlin, Rasool Fakoor. Replay-based recurrent RL for task-agnostic agents, evaluated on a synthetic setting and Meta-World. Recurrence addresses hidden-context inference, which must be separated experimentally from Bayesian weight consolidation.
Source: https://arxiv.org/abs/2205.14495

## CORA
Sam Powers et al., CoLLAs 2022. Platform with Atari, Procgen, NetHack, and CHORES benchmark families plus continual evaluation, isolated forgetting, and zero-shot forward transfer metrics.
Source: https://arxiv.org/abs/2110.10067
Code: https://github.com/AGI-Labs/continual_rl

## ContinualBench
sail-sg/ContinualBench provides a continual-RL environment with unified world dynamics. Its repository cites the ICML 2025 paper "Continual Reinforcement Learning by Planning with Online World Models" and states an MIT license.
Source: https://github.com/sail-sg/ContinualBench

## Audit backlog
Before the comparison suite is complete, expand with verified records for FOO-VB as used by BGD, EWC/Online EWC, SI, MAS, LwF where relevant, MESU/controlled-forgetting Bayesian work, CARL, current Continual World methods, generalized Bayesian policy learning, posterior/Thompson sampling in RL, plasticity-loss work, modern world-model CRL, replay evidence-reuse literature, and code licenses for any implementation actually reused.
