"""Boundary-free adaptive-BGD SAC run on a recurring synthetic control stream."""

from __future__ import annotations

import json
from dataclasses import asdict

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACAgent, BGDSACConfig
from rl_bgd.agents.sac.train import SACTrainConfig, train_sac
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.envs.synthetic.nonstationary_lqr import ScheduledLQREnv
from rl_bgd.metrics.change_detection import change_detection_metrics
from rl_bgd.surprise.base import EMANormalizerConfig, RetentionMappingConfig
from rl_bgd.surprise.td import AdaptiveTDRetentionConfig, TDSurpriseConfig
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


def run_adaptive_bgd_lqr_stream(
    *,
    total_steps: int = 900,
    phase_steps: int = 300,
    seed: int = 23,
    device: str = "auto",
    detection_threshold: float = 2.0,
) -> dict[str, object]:
    """Run A->B->A dynamics without passing phase boundaries to the agent."""

    if total_steps < 128 or phase_steps < 1:
        raise ValueError("training horizon is too short")
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="recurring",
            anchors=(
                {"dynamics": 0.9, "control_gain": 0.5},
                {"dynamics": 0.55, "control_gain": 0.25},
            ),
            phase_steps=phase_steps,
        )
    )
    env = ScheduledLQREnv(
        LinearQuadraticControlEnv(
            horizon=30,
            device=resolved,
        ),
        schedule,
    )
    adaptive = AdaptiveTDRetentionConfig(
        surprise=TDSurpriseConfig(
            aggregation="median_abs",
            normalizer=EMANormalizerConfig(
                decay=0.99,
                smoothing_decay=0.9,
                initial_variance=1.0,
            ),
        ),
        mapping=RetentionMappingConfig(
            lambda_min=0.5,
            kappa=1.0,
        ),
    )
    agent = BGDSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(32, 32),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        bgd_config=BGDSACConfig(
            bayesianization="critic_only",
            posterior_std=0.1,
            adaptive_td_retention=adaptive,
            critic_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
        ),
        device=resolved,
    )
    surprise_timeline: list[dict[str, float]] = []
    detected_steps: list[int] = []

    def observe_update(
        step: int,
        metrics: dict[str, float],
    ) -> None:
        if "surprise_normalized" not in metrics:
            return
        record = {
            "step": float(step),
            "surprise_normalized": metrics["surprise_normalized"],
            "surprise_smoothed": metrics["surprise_smoothed"],
            "retention_lambda": metrics["retention_lambda"],
            "critic1_sigma_mean": metrics["critic1_sigma_mean"],
            "critic1_effective_lr_mean": metrics[
                "critic1_effective_lr_mean"
            ],
        }
        surprise_timeline.append(record)
        if metrics["surprise_normalized"] >= detection_threshold:
            detected_steps.append(step)

    training = train_sac(
        env,
        agent,
        config=SACTrainConfig(
            total_steps=total_steps,
            random_steps=64,
            batch_size=64,
            replay_capacity=max(2_000, total_steps),
            seed=seed,
        ),
        update_observer=observe_update,
    )
    true_changes = list(
        range(
            phase_steps,
            total_steps,
            phase_steps,
        )
    )
    detection = change_detection_metrics(
        true_changes,
        detected_steps,
        tolerance_steps=max(phase_steps // 3, 1),
        total_steps=total_steps,
    )
    return {
        "training": training,
        "true_change_steps": true_changes,
        "detected_steps": detected_steps,
        "change_detection": asdict(detection),
        "surprise_timeline": surprise_timeline,
        "information_access": {
            "receives_task_id": False,
            "receives_task_boundary": False,
            "context_available": False,
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_adaptive_bgd_lqr_stream(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
