"""Soft Actor-Critic implementations."""

from rl_bgd.agents.sac.agent import SACAgent, SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACAgent, BGDSACConfig
from rl_bgd.agents.sac.task_aware_agent import TaskAwareSACAgent
from rl_bgd.agents.sac.task_aware_train import (
    CanonicalSACTrainConfig,
    train_canonical_task_aware_sac,
)
from rl_bgd.agents.sac.recurrent_agent import (
    RecurrentSACAgent,
    RecurrentSACConfig,
)
from rl_bgd.agents.sac.recurrent_bgd_agent import (
    BGDRecurrentSACAgent,
)
from rl_bgd.agents.sac.recurrent_train import (
    RecurrentSACTrainConfig,
    evaluate_recurrent_sac,
    train_recurrent_sac,
)

__all__ = [
    "BGDRecurrentSACAgent",
    "BGDSACAgent",
    "BGDSACConfig",
    "RecurrentSACAgent",
    "RecurrentSACConfig",
    "RecurrentSACTrainConfig",
    "SACAgent",
    "SACConfig",
    "TaskAwareSACAgent",
    "CanonicalSACTrainConfig",
    "evaluate_recurrent_sac",
    "train_recurrent_sac",
    "train_canonical_task_aware_sac",
]
