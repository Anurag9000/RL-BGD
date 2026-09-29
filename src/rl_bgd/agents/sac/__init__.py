"""Soft Actor-Critic implementation."""

from rl_bgd.agents.sac.agent import SACAgent, SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACAgent, BGDSACConfig

__all__ = [
    "BGDSACAgent",
    "BGDSACConfig",
    "SACAgent",
    "SACConfig",
]
