"""Serializable run identities, signed manifests, and deterministic plan creation."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from .config import RunnerSettings
from .datasets import DatasetProvider, DatasetTaskRef
from .route import ModelRoute
from ..registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from .models import ModelAdapter


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def manifest_compatibility_digest(manifest: dict[str, Any]) -> str:
    """Hash the full manifest, normalizing only absolute interpreter aliases."""
    payload = json.loads(canonical_json(manifest))
    payload.pop("manifest_digest", None)
    for job in payload.get("jobs", []):
        command = job.get("command") if isinstance(job, dict) else None
        if isinstance(command, list) and command and isinstance(command[0], str) and os.path.isabs(command[0]):
            command[0] = os.path.realpath(command[0])
    return sha256_json(payload)


def derive_run_seed(base_seed: int, identity_fields: Iterable[object]) -> int:
    identity = "|".join([str(int(base_seed)), *(str(value) for value in identity_fields)])
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return seed if seed > 0 else max(1, int(base_seed) % (2**31 - 1))


@dataclass(frozen=True, slots=True)
class SplitSpec:
    repeat: int
    fold: int
    sample: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    split_sha256: str

    def save(self, path: Path) -> None:
        actual_hash = _sha_indices(self.train_indices, self.validation_indices, self.test_indices)
        if actual_hash != self.split_sha256:
            raise ValueError(
                "split signature does not match its train/validation/test indices"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = np.load(path, allow_pickle=False)
            current_hash = str(existing["split_sha256"].item())
            if current_hash != self.split_sha256:
                raise ValueError(f"existing split archive has a different signature: {path}")
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                train=np.asarray(self.train_indices, dtype=np.int64),
                validation=np.asarray(self.validation_indices, dtype=np.int64),
                test=np.asarray(self.test_indices, dtype=np.int64),
                split_sha256=np.asarray(self.split_sha256),
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, *, repeat: int, fold: int, sample: int) -> "SplitSpec":
        with np.load(path, allow_pickle=False) as archive:
            train = tuple(int(value) for value in archive["train"])
            validation = tuple(int(value) for value in archive["validation"])
            test = tuple(int(value) for value in archive["test"])
            signature = str(archive["split_sha256"].item())
        actual = _sha_indices(train, validation, test)
        if actual != signature:
            raise ValueError(f"split archive checksum mismatch: {path}")
        if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
            raise ValueError(f"split archive contains overlapping partitions: {path}")
        return cls(repeat, fold, sample, train, validation, test, signature)


def _sha_indices(*arrays: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(tuple(array), dtype=np.int64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RunIdentity:
    provider: str
    suite: str
    task_id: str
    split_repeat: int
    fold: int
    sample: int
    suite_repeat: int
    model_route: str
    seed: int
    split_sha256: str
    run_signature: str

    @property
    def run_id(self) -> str:
        def safe(value: object) -> str:
            text = "".join(ch if ch.isalnum() or ch in "-_@." else "_" for ch in str(value))
            return text.strip("._") or "unknown"

        stable = {
            "provider": self.provider,
            "suite": self.suite,
            "task_id": self.task_id,
            "split_repeat": self.split_repeat,
            "fold": self.fold,
            "sample": self.sample,
            "suite_repeat": self.suite_repeat,
            "model_route": self.model_route,
            "seed": self.seed,
            "split_sha256": self.split_sha256,
            "run_signature": self.run_signature,
        }
        digest = sha256_json(stable)[:16]
        return f"{safe(self.provider)}/{safe(self.suite)}/task-{safe(self.task_id)}/split-r{self.split_repeat}-f{self.fold}-s{self.sample}/suite-r{self.suite_repeat}/{safe(self.model_route)}/seed-{self.seed}-{digest}"

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["run_id"] = self.run_id
        return result


@dataclass(frozen=True, slots=True)
class RunSpec:
    identity: RunIdentity
    task_ref: DatasetTaskRef
    model_adapter: str
    route: ModelRoute
    split_file: str
    result_dir: str
    worker_settings: dict[str, object]
    command: tuple[str, ...]
    device_capable: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.as_dict(),
            "task_ref": asdict(self.task_ref),
            "model_adapter": self.model_adapter,
            "route": self.route.as_dict(),
            "split_file": self.split_file,
            "result_dir": self.result_dir,
            "worker_settings": self.worker_settings,
            "command": list(self.command),
            "device_capable": self.device_capable,
        }


@dataclass(frozen=True, slots=True)
class SuitePlan:
    run_signature: str
    manifest: dict[str, Any]
    runs: tuple[RunSpec, ...]

    @classmethod
    def from_manifest(cls, path: str | Path) -> "SuitePlan":
        from .route import ModelRoute

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        signature = str(payload["run_signature"])
        expected_digest = payload.get("manifest_digest")
        if expected_digest is not None and str(expected_digest) != manifest_compatibility_digest(payload):
            raise ValueError("run manifest digest is invalid")
        runs: list[RunSpec] = []
        for raw in payload.get("jobs", []):
            identity_value = dict(raw["identity"])
            identity_value.pop("run_id", None)
            identity = RunIdentity(**identity_value)
            task_ref = DatasetTaskRef(**raw["task_ref"])
            route_value = dict(raw["route"])
            route_value.pop("route_label", None)
            fixed = route_value.get("fixed_grid")
            if fixed is not None:
                route_value["fixed_grid"] = tuple(tuple(int(width) for width in vector) for vector in fixed)
            route = ModelRoute(**route_value)
            runs.append(RunSpec(
                identity=identity,
                task_ref=task_ref,
                model_adapter=str(raw["model_adapter"]),
                route=route,
                split_file=str(raw["split_file"]),
                result_dir=str(raw["result_dir"]),
                worker_settings=dict(raw["worker_settings"]),
                command=tuple(raw["command"]),
                device_capable=bool(raw.get("device_capable", True)),
            ))
        return cls(signature, payload, tuple(runs))

    @classmethod
    def build(
        cls,
        *,
        settings: RunnerSettings,
        results_dir: str | Path,
        run_root: str | Path,
        dataset_providers: tuple[DatasetProvider, ...],
        dataset_suites: tuple[tuple[str, str], ...],
        model_adapters: tuple[ModelAdapter, ...],
        generic_profiles: tuple[str, ...] | None = None,
        model_seeds: tuple[int, ...] | None = None,
        validation_fraction: float | None = None,
    ) -> "SuitePlan":
        from .route import ModelRoute

        generic_profiles = generic_profiles or settings.generic_profiles
        model_seeds = model_seeds or settings.model_seeds
        validation_fraction = settings.validation_fraction if validation_fraction is None else validation_fraction
        if not model_seeds:
            raise ValueError("at least one model seed is required")
        if not 0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and one half")
        if not dataset_suites or not dataset_providers or not model_adapters:
            raise ValueError("dataset suites, providers, and model adapters are required")
        result_root = Path(results_dir).resolve()
        runtime_root = Path(run_root).resolve()
        provider_by_name = {provider.name: provider for provider in dataset_providers}
        adapter_routes = [
            (adapter, route)
            for adapter in model_adapters
            for route in adapter.enumerate_routes(generic_profiles, settings)
        ]
        requested_tasks = set(settings.tasks)
        if not requested_tasks or "all" in requested_tasks:
            requested_tasks = {"width_only", "width_to_depth", "depth_to_width", "depth_only"}
        if settings.surface == "active":
            requested_tasks.intersection_update({"width_only", "width_to_depth", "depth_to_width", "depth_only"})
        elif settings.surface == "backlog":
            requested_tasks.clear()
        route_groups = {
            "width_only": settings.include_width_only,
            "width_to_depth": settings.include_width_to_depth,
            "depth_to_width": settings.include_depth_to_width,
            "depth_only": settings.include_depth_only,
        }

        def route_group(route: ModelRoute) -> str:
            if route.screening_only:
                return "screening"
            if route.suite_group is not None:
                allowed = {"width_only", "width_to_depth", "depth_to_width", "depth_only", "vanilla", "screening"}
                if route.suite_group not in allowed:
                    raise ValueError(f"model route {route.route_label!r} has unsupported suite group {route.suite_group!r}")
                return route.suite_group
            if route.model_id == "MLP-Fixed":
                return "vanilla"
            config = MODEL_REGISTRY.get(route.model_id)
            if config is None:
                raise ValueError(
                    f"custom model route {route.model_id!r} must declare suite_group explicitly"
                )
            return {
                "width": "width_only",
                "width_then_depth": "width_to_depth",
                "depth_then_width": "depth_to_width",
                "depth": "depth_only",
            }.get(config.order, "unknown")

        adapter_routes = [
            (adapter, route)
            for adapter, route in adapter_routes
            if (
                route_group(route) == "screening"
                and settings.run_hardness_screen
                and settings.surface != "backlog"
            ) or (
                route_group(route) == "vanilla"
                and settings.include_vanilla
                and settings.surface != "backlog"
            ) or (
                route_group(route) in requested_tasks
                and route_groups.get(route_group(route), False)
            )
        ]
        routes = [route for _adapter, route in adapter_routes]
        provider_suites: list[tuple[DatasetProvider, str]] = []
        for provider_name, suite in dataset_suites:
            if provider_name not in provider_by_name:
                raise ValueError(f"dataset provider {provider_name!r} was not registered")
            provider_suites.append((provider_by_name[provider_name], suite))

        task_inputs: list[tuple[DatasetProvider, DatasetTaskRef, Any, list[SplitSpec]]] = []
        for provider, suite in provider_suites:
            refs = provider.list_tasks(suite)
            if settings.task_limit is not None:
                refs = refs[:settings.task_limit]
            if not refs:
                raise ValueError(f"dataset provider {provider.name!r} returned no tasks for suite {suite!r}")
            for ref in refs:
                data = provider.load_task(ref)
                source_metadata = getattr(data, "dataset_metadata", {})
                if source_metadata is None:
                    source_metadata = {}
                if not isinstance(source_metadata, Mapping):
                    raise TypeError(
                        f"dataset provider {provider.name!r} returned dataset_metadata "
                        "that is not a mapping"
                    )
                ref = replace(
                    ref,
                    dataset_id=str(getattr(data, "dataset_id", ref.dataset_id)) if getattr(data, "dataset_id", ref.dataset_id) is not None else None,
                    metadata={
                        **(ref.metadata or {}),
                        "dataset_name": getattr(data, "name", None),
                        "target_name": getattr(data, "target_name", None),
                        "content_sha256": getattr(data, "content_sha256", None),
                        "rows": len(data.X) if hasattr(data, "X") else None,
                        "raw_features": int(data.X.shape[1]) if hasattr(data, "X") else None,
                        "dataset_metadata": dict(source_metadata),
                    },
                )
                split_settings = type("SplitSettings", (), {
                    "seed": settings.seed,
                    "validation_fraction": validation_fraction,
                    "fold_limit": settings.fold_limit,
                })()
                splits = provider.list_splits(ref, data, split_settings)
                task_inputs.append((provider, ref, data, splits))

        signature_payload = {
            "protocol_version": 1,
            "runner_settings": settings.as_dict(),
            "providers_and_suites": dataset_suites,
            "model_adapters": [adapter.name for adapter in model_adapters],
            "routes": [route.as_dict() for route in routes],
            "effective_repeat_count": max(settings.repeat_count, settings.prediction_repeats, settings.missing_present_task_repeats, 1),
            "model_seeds": model_seeds,
            "validation_fraction": validation_fraction,
            "results_dir": str(result_root),
            "run_root": str(runtime_root),
            "python_executable": os.path.realpath(sys.executable),
            "split_identities": [
                {
                    "provider": provider.name,
                    "suite": ref.suite,
                    "task_id": ref.task_id,
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "sample": split.sample,
                    "split_sha256": split.split_sha256,
                }
                for provider, ref, _data, splits in task_inputs
                for split in splits
            ],
            "dataset_identities": [
                {
                    "provider": provider.name,
                    "suite": ref.suite,
                    "task_id": ref.task_id,
                    "dataset_id": ref.dataset_id,
                    "content_sha256": getattr(data, "content_sha256", None),
                "metadata": ref.metadata,
                }
                for provider, ref, data, _ in task_inputs
            ],
        }
        signature = sha256_json(signature_payload)
        run_specs: list[RunSpec] = []
        has_adaptive = any(route_group(route) not in {"vanilla", "screening"} for route in routes)
        outer_repeat_count = max(settings.repeat_count, settings.missing_present_task_repeats if has_adaptive else 0, 1)
        for suite_repeat in range(1, outer_repeat_count + 1):
            for provider, ref, data, splits in task_inputs:
                # Preserve the recorded asymmetric repeat dispatch: ADP routes
                # use missing_present_task_repeats; fixed vanilla routes use
                # every outer repeat.
                for adapter, route in adapter_routes:
                    route_is_vanilla = route_group(route) == "vanilla" and not route.screening_only
                    route_repeat_limit = (
                        outer_repeat_count if route_is_vanilla else
                        1 if route.screening_only else
                        max(1, settings.missing_present_task_repeats or settings.repeat_count)
                    )
                    if suite_repeat > route_repeat_limit:
                        continue
                    for split in splits:
                        split_path = result_root / "splits" / ref.provider / ref.suite / (
                            f"task-{ref.task_id}-r{split.repeat}-f{split.fold}-s{split.sample}-{signature[:12]}.npz"
                        )
                        split.save(split_path)
                        effective_model_seeds = (0,) if route.screening_only else model_seeds
                        for model_seed in effective_model_seeds:
                            seed = route.screening_seed if route.screening_only else derive_run_seed(
                                settings.seed,
                                (ref.provider, ref.suite, ref.task_id, split.repeat, split.fold, split.sample,
                                 suite_repeat, route.route_label, model_seed),
                            )
                            identity = RunIdentity(
                                provider=ref.provider,
                                suite=ref.suite,
                                task_id=ref.task_id,
                                split_repeat=split.repeat,
                                fold=split.fold,
                                sample=split.sample,
                                suite_repeat=suite_repeat,
                                model_route=route.route_label,
                                seed=seed,
                                split_sha256=split.split_sha256,
                                run_signature=signature,
                            )
                            job_dir = runtime_root / "jobs" / identity.run_id
                            result_dir = result_root / identity.run_id
                            spec_path = job_dir / "run_spec.json"
                            command = (sys.executable, "-m", "simple_adp.runner.worker", "--spec", str(spec_path))
                            spec = RunSpec(
                                identity=identity,
                                task_ref=ref,
                                model_adapter=adapter.name,
                                route=route,
                                split_file=str(split_path),
                                result_dir=str(result_dir),
                                worker_settings={
                                    "max_epochs": settings.max_epochs,
                                    "max_epochs_vanilla": settings.max_epochs_vanilla,
                                    "max_epochs_adp": settings.max_epochs_adp,
                                    "batch_size": settings.batch_size,
                                    "smoke_test": settings.smoke_test,
                                    "cpu_only": settings.cpu_only,
                                    "init_width": settings.width_to_depth_width,
                                    "init_depth": settings.width_to_depth_depth,
                                    "max_width": settings.max_width,
                                    "max_depth": settings.max_depth,
                                    "max_neurons": settings.max_neurons,
                                    "ex_k_width": settings.ex_k_width,
                                    "ex_k_depth": settings.ex_k_depth,
                                    "patience_es": settings.patience_es,
                                    "patience_width_exp": settings.patience_width_exp,
                                    "patience_depth_exp": settings.patience_depth_exp,
                                    "delta_width": settings.delta_width,
                                    "delta_depth": settings.delta_depth,
                                    "learning_rate": settings.learning_rate,
                                    "scheduler_t0": settings.scheduler_t0,
                                    "scheduler_t_mult": settings.scheduler_t_mult,
                                    "scheduler_eta_min": settings.scheduler_eta_min,
                                    "weight_decay": settings.weight_decay,
                                    "grad_clip": settings.grad_clip,
                                    "lambda_loss": settings.lambda_loss,
                                    "lambda_eq": settings.lambda_eq,
                                    "lambda_ineq": settings.lambda_ineq,
                                    "max_candidates": settings.max_candidates,
                                    "min_new_layer_width": settings.min_new_layer_width,
                                    "seed": seed,
                                    "validation_fraction": validation_fraction,
                                    "device": "cpu" if settings.cpu_only else "cuda",
                                    "screening": "diagnostic_only",
                                },
                                command=command,
                            )
                            spec_path.parent.mkdir(parents=True, exist_ok=True)
                            if not spec_path.exists():
                                tmp = spec_path.with_suffix(".tmp")
                                tmp.write_text(json.dumps(spec.as_dict(), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
                                tmp.replace(spec_path)
                            run_specs.append(spec)
        selected = run_specs[max(0, settings.job_start_index):]
        if settings.job_limit > 0:
            selected = selected[: settings.job_limit]
        manifest = {
            **signature_payload,
            "run_signature": signature,
            "results_dir": str(result_root),
            "run_root": str(runtime_root),
            "python_executable": os.path.realpath(sys.executable),
            "job_count": len(selected),
            "jobs": [spec.as_dict() for spec in selected],
        }
        manifest["manifest_digest"] = manifest_compatibility_digest(manifest)
        return cls(signature, manifest, tuple(selected))

    def write_manifest(self, run_root: str | Path) -> Path:
        path = Path(run_root) / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = None
            expected_digest = manifest_compatibility_digest(self.manifest)
            existing_signature = existing.get("run_signature") if isinstance(existing, dict) else None
            existing_digest = manifest_compatibility_digest(existing) if isinstance(existing, dict) else None
            if existing_signature != self.run_signature or existing_digest != expected_digest:
                mismatch = path.with_name("manifest_mismatch.json")
                mismatch.write_text(json.dumps({
                    "expected_signature": self.run_signature,
                    "found_signature": existing_signature,
                    "expected_digest": expected_digest,
                    "found_digest": existing_digest,
                }, indent=2) + "\n", encoding="utf-8")
                raise ValueError("run manifest does not match this suite plan")
            return path
        payload = dict(self.manifest)
        payload["manifest_digest"] = manifest_compatibility_digest(payload)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path
