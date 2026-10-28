"""Generic runner configuration with the ledger's recorded defaults."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

UNBOUNDED_CONCURRENCY = 10**9
SchedulerLabel = Literal["pressure_aware", "gpu_first", "fixed"]


def parse_epoch_limit(value: int | str) -> int:
    """Parse finite epochs; the ledger's unlimited spellings map to zero."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"inf", "+inf", "infinity", "+infinity", "unlimited", "none", "0"}:
            return 0
        try:
            parsed = int(normalized)
        except ValueError as exc:
            raise ValueError(f"invalid epoch limit {value!r}") from exc
    else:
        parsed = int(value)
    if parsed < 0:
        raise ValueError("epoch limit cannot be negative")
    return parsed


def _int_tuple(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            return tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(f"invalid integer grid {value!r}") from exc
    return tuple(int(item) for item in value)


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    """Control-plane settings, intentionally separate from model search settings."""

    concurrency: int = 0
    concurrency_file: str | None = None
    max_active_jobs: int = 0
    max_active_gpu_jobs: int = 0
    repeat_count: int = 5
    missing_present_task_repeats: int = 1
    prediction_repeats: int = 1
    max_retries: int = 1

    host_ram_stop_avail_mib: float = 1024.0
    host_ram_resume_avail_mib: float = 1280.0
    gpu_stop_avail_mib: float = 100.0
    gpu_resume_avail_mib: float = 500.0
    launch_host_reserve_mib: float = 0.0
    launch_gpu_reserve_mib: float = 0.0
    child_host_gate_buffer_mib: float = 500.0
    child_gpu_gate_buffer_mib: float = 100.0
    external_memory_drop_mib: float = 500.0
    pressure_poll_seconds: float = 0.5
    post_launch_sample_delay_seconds: float = 30.0
    batch_backoff_factor: float = 0.5
    gpu_device_index: int = 0
    cpu_only: bool = False
    scheduler: SchedulerLabel = "gpu_first"
    swap_pressure_pct: float = 70.0
    swap_resume_pct: float = 70.0

    max_width: int = 512
    max_depth: int = 5
    max_neurons: int = 1_000_000_000
    # Estimator/search controls are serialized with each run spec and passed
    # to the worker. Keep their defaults aligned with SearchSettings.
    ex_k_width: int = 1
    ex_k_depth: int = 1
    patience_es: int = 10
    patience_width_exp: int = 10
    patience_depth_exp: int = 10
    delta_width: float = 0.0
    delta_depth: float = 0.0
    learning_rate: float = 1e-3
    scheduler_t0: int = 10
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float | None = None
    lambda_loss: float = 1.0
    lambda_eq: float = 0.0
    lambda_ineq: float = 0.0
    max_candidates: int = 0
    min_new_layer_width: int = 1
    max_epochs_vanilla: int = 0
    max_epochs_adp: int = 0
    max_epochs: int | None = None
    batch_size: int = 128
    smoke_test: bool = False
    task_limit: int | None = None
    fold_limit: int | None = None
    model_seeds: tuple[int, ...] = (0, 1, 2)
    generic_profiles: tuple[str, ...] = ("faithful", "corrected")
    validation_fraction: float = 0.2
    screen_seed_count: int = 2
    classification_ceiling: float = 1.0
    regression_ceiling_r2: float = 0.999
    min_complexity_gain: float = 0.02
    min_hard_fold_fraction: float = 0.8
    run_hardness_screen: bool = True
    screen_widths: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    screen_depths: tuple[int, ...] = (2, 3, 5)
    deeper_screen_widths: tuple[int, ...] = (16, 64, 256, 512)
    seed: int = 42
    cuda_visible_devices: str = "0"
    pytorch_cuda_alloc_conf: str = "expandable_segments:True,max_split_size_mb:128"
    cublas_workspace_config: str = ":4096:8"
    shared_cpu: bool = False
    cpu_workers: int = 0

    mode: Literal["suite"] = "suite"
    surface: Literal["active", "backlog", "all"] = "all"
    tasks: tuple[str, ...] = ("all",)
    include_vanilla: bool = True
    include_width_only: bool = True
    include_width_to_depth: bool = True
    include_depth_to_width: bool = True
    include_depth_only: bool = True
    width_only_depths: tuple[int, ...] = (1, 2, 3, 4, 5)
    width_only_width: int = 1
    width_to_depth_width: int = 1
    width_to_depth_depth: int = 1
    # Source compatibility field: parsed/validated/serialized by the ledger
    # runner, but combined-growth jobs use the scalar width/depth seeds below.
    width_to_depth_widths: tuple[int, ...] = (1,)
    depth_only_widths: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    vanilla_depths: tuple[int, ...] = (1, 2, 3, 4, 5)
    vanilla_widths: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    vanilla_min_band: int = 1
    vanilla_max_band: int = 6
    vanilla_samples: int = 3
    job_start_index: int = 0
    job_limit: int = 0
    dry_run: bool = False
    results_dir: str = "benchmark-results/coordinated-suite"
    run_root: str | None = None
    source_run_root: str | None = None
    auto_resume_bundle: bool = True

    def __post_init__(self) -> None:
        for name in (
            "width_only_depths", "width_to_depth_widths", "depth_only_widths",
            "vanilla_depths", "vanilla_widths", "screen_widths", "screen_depths", "deeper_screen_widths",
        ):
            object.__setattr__(self, name, _int_tuple(getattr(self, name)))
        object.__setattr__(self, "model_seeds", _int_tuple(self.model_seeds))
        if isinstance(self.generic_profiles, str):
            object.__setattr__(self, "generic_profiles", tuple(part.strip() for part in self.generic_profiles.split(",") if part.strip()))
        else:
            object.__setattr__(self, "generic_profiles", tuple(self.generic_profiles))
        for name in ("max_epochs_vanilla", "max_epochs_adp"):
            object.__setattr__(self, name, parse_epoch_limit(getattr(self, name)))
        if self.max_epochs is not None:
            object.__setattr__(self, "max_epochs", parse_epoch_limit(self.max_epochs))
        self.validate()

    def validate(self) -> None:
        nonnegative = (
            "concurrency", "gpu_device_index", "max_retries", "repeat_count",
            "missing_present_task_repeats", "max_active_jobs", "max_active_gpu_jobs",
            "launch_host_reserve_mib", "launch_gpu_reserve_mib",
            "child_host_gate_buffer_mib", "child_gpu_gate_buffer_mib",
        )
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.missing_present_task_repeats < 1:
            raise ValueError("missing_present_task_repeats must be >=1")
        if self.pressure_poll_seconds <= 0:
            raise ValueError("pressure_poll_seconds must be positive")
        if self.external_memory_drop_mib <= 0:
            raise ValueError("external_memory_drop_mib must be positive")
        if self.host_ram_resume_avail_mib < self.host_ram_stop_avail_mib:
            raise ValueError("host resume threshold must be >= stop threshold")
        if self.gpu_resume_avail_mib < self.gpu_stop_avail_mib:
            raise ValueError("GPU resume threshold must be >= stop threshold")
        if self.post_launch_sample_delay_seconds < 0:
            raise ValueError("post-launch sample delay cannot be negative")
        if min(self.max_width, self.max_depth, self.max_neurons) <= 0:
            raise ValueError("architecture capacity limits must be positive")
        if min(self.ex_k_width, self.ex_k_depth, self.patience_es,
               self.scheduler_t0, self.scheduler_t_mult, self.min_new_layer_width) <= 0:
            raise ValueError("estimator increments, epoch patience, scheduler values, and depth seed must be positive")
        if min(self.patience_width_exp, self.patience_depth_exp, self.max_candidates) < 0:
            raise ValueError("architecture patience and max_candidates cannot be negative")
        if min(self.delta_width, self.delta_depth, self.weight_decay,
               self.scheduler_eta_min, self.lambda_loss, self.lambda_eq,
               self.lambda_ineq) < 0 or self.learning_rate <= 0:
            raise ValueError("invalid estimator optimizer or search values")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive when set")
        if not 0 < self.batch_backoff_factor < 1:
            raise ValueError("batch_backoff_factor must be strictly between zero and one")
        if self.scheduler not in {"pressure_aware", "gpu_first", "fixed"}:
            raise ValueError("unknown scheduler label")
        if self.prediction_repeats < 1:
            raise ValueError("prediction_repeats must be >=1")
        if self.surface not in {"active", "backlog", "all"}:
            raise ValueError("surface must be active, backlog, or all")
        if self.cpu_workers != 0 or self.shared_cpu:
            # The runner contract passes partitioned CPU mode and zero workers.
            raise ValueError("ledger-compatible child CPU settings require shared_cpu=False and cpu_workers=0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.task_limit is not None and self.task_limit <= 0:
            raise ValueError("task_limit must be positive")
        if self.fold_limit is not None and self.fold_limit <= 0:
            raise ValueError("fold_limit must be positive")
        if not self.model_seeds:
            raise ValueError("at least one model seed is required")
        if not self.generic_profiles or set(self.generic_profiles) - {"faithful", "corrected"}:
            raise ValueError("generic_profiles may contain faithful and/or corrected")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and one half")
        if self.screen_seed_count <= 0 or self.min_complexity_gain < 0:
            raise ValueError("invalid screening settings")
        if not 0 < self.min_hard_fold_fraction <= 1:
            raise ValueError("min_hard_fold_fraction must be in (0, 1]")
        if not 0 <= self.classification_ceiling <= 1:
            raise ValueError("classification_ceiling must be between zero and one")
        for name in (
            "width_only_depths", "width_to_depth_widths", "depth_only_widths",
            "vanilla_depths", "vanilla_widths", "screen_widths", "screen_depths", "deeper_screen_widths",
        ):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} cannot be empty")
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} values must be positive")
        if any(value > self.max_width for value in (*self.width_to_depth_widths, *self.depth_only_widths, *self.vanilla_widths)):
            raise ValueError("width grid exceeds max_width")
        if not 1 <= self.width_only_width <= self.max_width:
            raise ValueError("width_only_width must be between 1 and max_width")
        if not 1 <= self.width_to_depth_width <= self.max_width:
            raise ValueError("width_to_depth_width must be between 1 and max_width")
        if not 1 <= self.width_to_depth_depth <= self.max_depth:
            raise ValueError("width_to_depth_depth must be between 1 and max_depth")
        if any(value > self.max_depth for value in (*self.width_only_depths, *self.vanilla_depths)):
            raise ValueError("depth grid exceeds max_depth")
        if any(value > self.max_depth for value in self.screen_depths):
            raise ValueError("screen depth grid exceeds max_depth")
        if any(value > self.max_width for value in (*self.screen_widths, *self.deeper_screen_widths)):
            raise ValueError("screen width grid exceeds max_width")
        # As in the ledger, seed sign, stop-threshold signs, swap percentages,
        # and job slicing values are deliberately not checked here.

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def effective_concurrency(self) -> int:
        return UNBOUNDED_CONCURRENCY if self.concurrency == 0 else self.concurrency

    def resolve_concurrency(self) -> int:
        """Read the optional control file every call; malformed values are fatal."""
        configured = self.concurrency
        if self.concurrency_file:
            try:
                raw = Path(self.concurrency_file).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ValueError(f"cannot read concurrency file {self.concurrency_file!r}") from exc
            if raw:
                try:
                    configured = int(raw)
                except ValueError as exc:
                    raise ValueError("concurrency file must contain a non-negative integer") from exc
                if configured < 0:
                    raise ValueError("concurrency file must contain a non-negative integer")
        return UNBOUNDED_CONCURRENCY if configured == 0 else configured

    def launch_budget(self, batch_scale: float = 1.0) -> int:
        return max(1, math.ceil(self.resolve_concurrency() * batch_scale))


@dataclass(frozen=True, slots=True)
class StandalonePressureSettings:
    """Percent-based controls for the standalone pause/continue wrapper."""

    host_pressure_pct: float = 90.0
    host_resume_pct: float = 85.0
    gpu_pressure_pct: float = 90.0
    gpu_resume_pct: float = 85.0
    external_memory_drop_mib: float = 500.0
    poll_seconds: float = 0.5
    swap_pressure_pct: float = 70.0
    swap_resume_pct: float = 70.0

    def __post_init__(self) -> None:
        for prefix in ("host", "gpu"):
            pressure = getattr(self, f"{prefix}_pressure_pct")
            resume = getattr(self, f"{prefix}_resume_pct")
            if not 0 <= resume <= pressure <= 100:
                raise ValueError(f"{prefix} requires 0 <= resume <= pressure <= 100")
        if self.external_memory_drop_mib <= 0 or self.poll_seconds <= 0:
            raise ValueError("external drop and poll interval must be positive")
