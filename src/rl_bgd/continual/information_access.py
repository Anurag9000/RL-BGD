"""Central declaration of information available to an algorithm."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InformationAccessConfig:
    """Machine-checkable information-access contract for fair CRL comparisons."""

    receives_task_id: bool = False
    receives_task_boundary: bool = False
    replay: bool = False
    replay_size: int = 0
    uses_task_balanced_replay: bool = False
    architecture_growth: bool = False
    task_specific_heads: bool = False
    context_available: bool = False
    recurrent: bool = False
    stores_old_raw_data: bool = False
    posterior_distribution: str = "none"

    def validate_strict_task_agnostic(self) -> None:
        violations: list[str] = []
        if self.receives_task_id:
            violations.append("receives_task_id")
        if self.receives_task_boundary:
            violations.append("receives_task_boundary")
        if self.uses_task_balanced_replay:
            violations.append("uses_task_balanced_replay")
        if self.task_specific_heads:
            violations.append("task_specific_heads")
        if self.context_available:
            violations.append("context_available")
        if violations:
            raise ValueError(
                "strict task-agnostic configuration exposes privileged information: "
                + ", ".join(violations)
            )
        if self.replay_size < 0:
            raise ValueError("replay_size cannot be negative")
        if self.replay_size and not self.replay:
            raise ValueError("replay_size > 0 requires replay=True")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
