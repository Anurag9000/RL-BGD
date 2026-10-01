from typing import Any

from rl_bgd.runners import evidence_temperature_sweep


def test_temperature_sweep_keeps_controls_matched(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_bgd_sac_lqr(
        *,
        steps: int,
        seed: int,
        device: str,
        bayesianization: str,
        evidence_temperature: float,
    ) -> dict[str, object]:
        calls.append(
            {
                "steps": steps,
                "seed": seed,
                "device": device,
                "bayesianization": bayesianization,
                "evidence_temperature": evidence_temperature,
            }
        )
        return {
            "pre_return": -2.0,
            "post_return": -1.0,
            "improvement": 1.0,
            "training": {
                "final_10_mean_return": -1.1,
                "last_update_metrics": {
                    "critic1_sigma_mean": 0.1,
                },
            },
        }

    monkeypatch.setattr(
        evidence_temperature_sweep,
        "run_bgd_sac_lqr",
        fake_run_bgd_sac_lqr,
    )

    result = (
        evidence_temperature_sweep.run_evidence_temperature_sweep(
            temperatures=(
                0.5,
                1.0,
                2.0,
            ),
            steps=123,
            seed=17,
            device="cpu",
            bayesianization="critic_only",
        )
    )

    assert [
        call[
            "evidence_temperature"
        ]
        for call in calls
    ] == [
        0.5,
        1.0,
        2.0,
    ]
    for call in calls:
        assert call["steps"] == 123
        assert call["seed"] == 17
        assert call["device"] == "cpu"
        assert (
            call["bayesianization"]
            == "critic_only"
        )

    controls = result[
        "matched_controls"
    ]
    assert isinstance(controls, dict)
    assert all(
        bool(value)
        for value in controls.values()
    )
    assert len(
        result["runs"]
    ) == 3
