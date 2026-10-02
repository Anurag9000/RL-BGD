import pytest
import torch

from rl_bgd.runners.continual_world_sac import run_ta_continual_world_sac


@pytest.mark.benchmark
def test_ta_cw10_runner_builds_complete_evaluation_matrix() -> None:
    pytest.importorskip("metaworld")
    result = run_ta_continual_world_sac(
        benchmark="CW10",
        optimizer="adam",
        steps_per_task=1,
        seed=101,
        device="cpu",
        evaluation_episodes=1,
        episode_horizon=1,
        hidden_dims=(16, 16),
        replay_capacity=16,
        batch_size=2,
        random_steps=10,
    )
    assert len(result["task_names"]) == 10
    assert len(result["return_matrix"]) == 10
    assert len(result["success_matrix"]) == 10
    assert all(len(row) == 10 for row in result["return_matrix"])
    assert all(len(row) == 10 for row in result["success_matrix"])


def test_ta_runner_always_closes_protocol_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    import rl_bgd.runners.continual_world_sac as runner
    from rl_bgd.envs.continual_world.evaluation import TaskEvaluation
    from rl_bgd.envs.synthetic.lqr import TensorBox

    class FakeTrainEnv:
        def __init__(self) -> None:
            self.observation_space = TensorBox(
                low=torch.tensor([-1.0]),
                high=torch.tensor([1.0]),
            )
            self.action_space = TensorBox(
                low=torch.tensor([-1.0]),
                high=torch.tensor([1.0]),
            )

    class FakeBundle:
        def __init__(self) -> None:
            self.train_env = FakeTrainEnv()
            self.evaluation_envs = (object(), object())
            self.task_names = ("task-a", "task-b")
            self.closed = False

        def close(self) -> None:
            self.closed = True

    bundle = FakeBundle()

    def fake_protocol(*args: object, **kwargs: object) -> FakeBundle:
        del args, kwargs
        return bundle

    def fake_evaluate(*args: object, **kwargs: object) -> tuple[TaskEvaluation, ...]:
        del args, kwargs
        return (
            TaskEvaluation(mean_return=1.0, success_rate=1.0),
            TaskEvaluation(mean_return=2.0, success_rate=1.0),
        )

    def fake_train(
        env: object,
        agent: object,
        *,
        config: object,
        post_step_observer: object,
        **kwargs: object,
    ) -> dict[str, object]:
        del env, config, kwargs
        assert callable(post_step_observer)
        post_step_observer(1, agent)
        post_step_observer(2, agent)
        return {"steps": 2}

    monkeypatch.setattr(runner, "make_continual_world_protocol", fake_protocol)
    monkeypatch.setattr(runner, "evaluate_tasks", fake_evaluate)
    monkeypatch.setattr(runner, "train_sac", fake_train)

    result = runner.run_ta_continual_world_sac(
        benchmark="CW10",
        optimizer="adam",
        steps_per_task=1,
        device="cpu",
        evaluation_episodes=1,
        hidden_dims=(4,),
        replay_capacity=2,
        batch_size=2,
        random_steps=2,
    )
    assert result["return_matrix"] == [[1.0, 2.0], [1.0, 2.0]]
    assert bundle.closed


def test_ta_runner_supports_boundary_free_regularized_sac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_bgd.runners.continual_world_sac as runner
    from rl_bgd.agents.sac.regularized_agent import (
        RegularizedSACAgent,
    )
    from rl_bgd.envs.continual_world.evaluation import (
        TaskEvaluation,
    )
    from rl_bgd.envs.synthetic.lqr import TensorBox

    class FakeTrainEnv:
        def __init__(self) -> None:
            self.observation_space = TensorBox(
                low=torch.tensor(
                    [-1.0]
                ),
                high=torch.tensor(
                    [1.0]
                ),
            )
            self.action_space = TensorBox(
                low=torch.tensor(
                    [-1.0]
                ),
                high=torch.tensor(
                    [1.0]
                ),
            )

    class FakeBundle:
        def __init__(self) -> None:
            self.train_env = FakeTrainEnv()
            self.evaluation_envs = (
                object(),
                object(),
            )
            self.task_names = (
                "task-a",
                "task-b",
            )
            self.closed = False

        def close(self) -> None:
            self.closed = True

    bundle = FakeBundle()

    def fake_protocol(
        *args: object,
        **kwargs: object,
    ) -> FakeBundle:
        del args, kwargs
        return bundle

    def fake_evaluate(
        *args: object,
        **kwargs: object,
    ) -> tuple[
        TaskEvaluation,
        ...,
    ]:
        del args, kwargs
        return (
            TaskEvaluation(
                mean_return=1.0,
                success_rate=1.0,
            ),
            TaskEvaluation(
                mean_return=2.0,
                success_rate=1.0,
            ),
        )

    def fake_train(
        env: object,
        agent: object,
        *,
        config: object,
        post_step_observer: object,
        **kwargs: object,
    ) -> dict[str, object]:
        del env, config, kwargs
        assert isinstance(
            agent,
            RegularizedSACAgent,
        )
        assert callable(
            post_step_observer
        )
        post_step_observer(
            1,
            agent,
        )
        post_step_observer(
            2,
            agent,
        )
        return {
            "steps": 2,
            "last_update_metrics": {
                "consolidation_count": 0.0,
            },
        }

    monkeypatch.setattr(
        runner,
        "make_continual_world_protocol",
        fake_protocol,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_tasks",
        fake_evaluate,
    )
    monkeypatch.setattr(
        runner,
        "train_sac",
        fake_train,
    )

    result = runner.run_ta_continual_world_sac(
        benchmark="CW10",
        optimizer="ewc",
        steps_per_task=1,
        device="cpu",
        evaluation_episodes=1,
        hidden_dims=(4,),
        replay_capacity=2,
        batch_size=2,
        random_steps=2,
        consolidation_interval_updates=7,
    )

    access = result[
        "information_access"
    ]
    assert access[
        "receives_task_id"
    ] is False
    assert access[
        "receives_task_boundary"
    ] is False
    assert access[
        "consolidation_trigger"
    ] == "fixed_optimizer_update_interval"
    assert access[
        "consolidation_interval_updates"
    ] == 7
    assert bundle.closed
