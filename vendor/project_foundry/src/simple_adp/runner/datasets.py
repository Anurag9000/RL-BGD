"""Dataset provider contracts and the built-in OpenML task provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..tasks import TaskAdapter
    from .plan import SplitSpec


@dataclass(frozen=True, slots=True)
class DatasetTaskRef:
    provider: str
    suite: str
    task_id: str
    dataset_id: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.suite}:{self.task_id}"


@runtime_checkable
class TabularTaskData(Protocol):
    """Data contract passed from providers to model adapters.

    Dataset and target metadata must be JSON-serializable so it can be signed
    into the run plan. ``task_adapter`` is optional for the built-in binary and
    regression tasks; custom tasks provide their adapter on the payload.
    """

    X: Any
    y: Any
    dataset_id: str | int
    name: str
    target_name: str
    feature_schema: Sequence[str] | None
    content_sha256: str
    dataset_metadata: Mapping[str, Any]
    task_kind: Literal["classification", "regression"] | str
    task_adapter: TaskAdapter | None


@runtime_checkable
class DatasetProvider(Protocol):
    """Provider contract for task identity, tabular data and split generation.

    ``dataset_metadata`` is part of :class:`TabularTaskData` and is copied into
    each run specification, alongside the stable dataset identity and split.
    """

    name: str

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]: ...

    def load_task(self, ref: DatasetTaskRef) -> TabularTaskData: ...

    def list_splits(self, ref: DatasetTaskRef, data: TabularTaskData, settings: Any) -> list[SplitSpec]: ...


class DatasetRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, DatasetProvider] = {}

    def register(self, provider: DatasetProvider) -> None:
        name = provider.name.strip()
        if not name:
            raise ValueError("dataset provider name must be nonempty")
        if name in self._providers:
            raise ValueError(f"dataset provider {name!r} is already registered")
        self._providers[name] = provider

    def get(self, name: str) -> DatasetProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"unregistered dataset provider {name!r}") from exc

    @property
    def providers(self):
        return MappingProxyType(self._providers)


class OpenMLDatasetProvider:
    """Adapter around the existing split-safe OpenML benchmark primitives."""

    name = "openml"

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        from ..benchmark import OPENML_SUITES, load_suite_tasks

        if suite not in OPENML_SUITES:
            raise ValueError(f"unknown OpenML suite {suite!r}")
        spec = OPENML_SUITES[suite]
        return [DatasetTaskRef(self.name, suite, str(task_id)) for task_id in load_suite_tasks(suite)]

    def load_task(self, ref: DatasetTaskRef) -> TabularTaskData:
        from ..benchmark import load_openml_task

        if ref.provider != self.name:
            raise ValueError("task provider does not match OpenML provider")
        return load_openml_task(ref.suite, int(ref.task_id))

    def list_splits(self, ref: DatasetTaskRef, data: TabularTaskData, settings: Any) -> list[SplitSpec]:
        from ..benchmark import make_fold_split
        from .plan import SplitSpec

        dimensions = tuple(int(value) for value in data.task.get_split_dimensions())
        repeats, folds, samples = dimensions
        split_specs: list[SplitSpec] = []
        for repeat in range(repeats):
            for fold in range(folds):
                for sample in range(samples):
                    seed = int(settings.seed) + repeat * 100_003 + fold * 1_009 + sample
                    split = make_fold_split(
                        data.task,
                        data.y,
                        data.suite,
                        repeat=repeat,
                        fold=fold,
                        sample=sample,
                        validation_fraction=float(getattr(settings, "validation_fraction", 0.2)),
                        split_seed=seed,
                    )
                    split_specs.append(
                        SplitSpec(
                            repeat=repeat,
                            fold=fold,
                            sample=sample,
                            train_indices=tuple(int(i) for i in split.train_indices),
                            validation_indices=tuple(int(i) for i in split.validation_indices),
                            test_indices=tuple(int(i) for i in split.test_indices),
                            split_sha256=split.split_sha256,
                        )
                    )
                    if settings.fold_limit is not None and len(split_specs) >= int(settings.fold_limit):
                        return split_specs
        return split_specs
