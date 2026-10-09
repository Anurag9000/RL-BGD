from typing import Any

import pytest

from rl_bgd.agents.ppo.recurrent_train import RecurrentPPOTrainConfig
from rl_bgd.agents.ppo.train import PPOTrainConfig
from rl_bgd.agents.sac.recurrent_train import RecurrentSACTrainConfig
from rl_bgd.agents.sac.regularized_train import BoundaryRegularizedSACTrainConfig
from rl_bgd.agents.sac.task_aware_train import CanonicalSACTrainConfig
from rl_bgd.agents.sac.train import SACTrainConfig
from rl_bgd.utils.config_validation import (
    config_boolean,
    config_nonnegative_integer,
    config_positive_integer,
)


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_config_positive_integer_rejects_coercible_nonintegers(value: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        config_positive_integer(value, name="budget")


@pytest.mark.parametrize("value", [True, 0.0, "0"])
def test_config_nonnegative_integer_rejects_coercible_nonintegers(value: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        config_nonnegative_integer(value, name="seed")


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_config_boolean_rejects_truthy_nonbooleans(value: object) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        config_boolean(value, name="flag")


@pytest.mark.parametrize(
    "config",
    [
        SACTrainConfig(total_steps=True),  # type: ignore[arg-type]
        RecurrentSACTrainConfig(random_steps=False),  # type: ignore[arg-type]
        PPOTrainConfig(rollout_steps=True),  # type: ignore[arg-type]
        RecurrentPPOTrainConfig(seed=False),  # type: ignore[arg-type]
        BoundaryRegularizedSACTrainConfig(
            total_steps=10,
            consolidation_steps=(True,),  # type: ignore[arg-type]
        ),
        CanonicalSACTrainConfig(
            total_steps=10,
            reset_buffer_on_task_change=1,  # type: ignore[arg-type]
        ),
    ],
)
def test_training_configs_reject_coerced_runtime_controls(config: Any) -> None:
    with pytest.raises(TypeError):
        config.validate()
