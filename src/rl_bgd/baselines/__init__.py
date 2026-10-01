"""Continual-learning baseline mechanisms."""

from rl_bgd.baselines.ewc import (
    EWCRegularizer,
    OnlineEWCRegularizer,
)
from rl_bgd.baselines.importance import (
    empirical_fisher_diagonal,
    mas_importance,
)
from rl_bgd.baselines.mas import (
    MASRegularizer,
)
from rl_bgd.baselines.si import (
    SynapticIntelligence,
)

__all__ = [
    "EWCRegularizer",
    "MASRegularizer",
    "OnlineEWCRegularizer",
    "SynapticIntelligence",
    "empirical_fisher_diagonal",
    "mas_importance",
]
