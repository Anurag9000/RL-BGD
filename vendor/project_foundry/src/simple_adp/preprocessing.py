"""Train-only, schema-preserving preprocessing for mixed tabular features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureColumn:
    name: str
    kind: str


class TabularPreprocessor:
    """Median/standardize numeric columns and one-hot encode categoricals."""

    UNKNOWN = ("__simple_adp_unknown__", "")
    MISSING = ("__simple_adp_missing__", "")

    def __init__(self, feature_schema: Mapping[str, str] | Sequence[str] | None = None) -> None:
        self.feature_schema = feature_schema
        self.columns_: tuple[FeatureColumn, ...] | None = None
        self.numeric_stats_: dict[str, tuple[float, float, float]] = {}
        self.category_maps_: dict[str, dict[tuple[str, str], int]] = {}
        self.output_names_: tuple[str, ...] = ()

    @staticmethod
    def _category_key(value: object) -> tuple[str, str]:
        if pd.isna(value):
            return TabularPreprocessor.MISSING
        if isinstance(value, np.generic):
            value = value.item()
        return (type(value).__qualname__, repr(value))

    def _to_frame(self, values, *, fitting: bool) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
            frame.columns = [str(column) for column in frame.columns]
        else:
            array = np.asarray(values)
            if array.ndim != 2:
                raise ValueError("features must be a two-dimensional array or DataFrame")
            frame = pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])])
        if frame.empty or len(frame) == 0:
            raise ValueError("features must contain at least one row and one column")
        if frame.columns.duplicated().any():
            raise ValueError("feature names must be unique")
        if not fitting:
            expected = [column.name for column in self.columns_ or ()]
            missing = [name for name in expected if name not in frame.columns]
            extra = [name for name in frame.columns if name not in expected]
            if missing or extra:
                raise ValueError(f"feature schema mismatch; missing={missing}, extra={extra}")
            frame = frame.loc[:, expected]
        return frame

    def _column_kinds(self, frame: pd.DataFrame, *, array_input: bool = False) -> tuple[FeatureColumn, ...]:
        schema = self.feature_schema
        columns = []
        for index, name in enumerate(frame.columns):
            if isinstance(schema, Mapping):
                kind = schema.get(name)
            elif schema is not None:
                if len(schema) != len(frame.columns):
                    raise ValueError("feature_schema length must match the number of columns")
                kind = schema[index]
            else:
                kind = None
            if kind is None:
                kind = "numeric" if array_input or pd.api.types.is_numeric_dtype(frame[name].dtype) else "categorical"
            if kind not in {"numeric", "categorical"}:
                raise ValueError(f"unsupported feature kind {kind!r} for column {name!r}")
            columns.append(FeatureColumn(str(name), kind))
        if isinstance(schema, Mapping):
            unknown = set(schema) - set(frame.columns)
            if unknown:
                raise ValueError(f"feature_schema contains unknown columns: {sorted(unknown)}")
        return tuple(columns)

    def fit(self, features) -> "TabularPreprocessor":
        array_input = not isinstance(features, pd.DataFrame)
        frame = self._to_frame(features, fitting=True)
        self.columns_ = self._column_kinds(frame, array_input=array_input)
        output_names: list[str] = []
        for column in self.columns_:
            values = frame[column.name]
            if column.kind == "numeric":
                numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
                numeric[~np.isfinite(numeric)] = np.nan
                finite = numeric[np.isfinite(numeric)]
                median = float(np.median(finite)) if finite.size else 0.0
                filled = np.where(np.isfinite(numeric), numeric, median)
                mean = float(filled.mean())
                scale = float(filled.std())
                if not np.isfinite(scale) or scale < 1e-12:
                    scale = 1.0
                self.numeric_stats_[column.name] = (median, mean, scale)
                output_names.append(column.name)
            else:
                keys = {self._category_key(value) for value in values.to_numpy()}
                normal = sorted(keys)
                mapping = {self.UNKNOWN: 0}
                mapping.update({key: i + 1 for i, key in enumerate(normal)})
                self.category_maps_[column.name] = mapping
                output_names.extend([f"{column.name}==__UNKNOWN__"])
                output_names.extend(f"{column.name}=={key[0]}:{key[1]}" for key in normal)
        self.output_names_ = tuple(output_names)
        return self

    def transform(self, features) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("fit must be called before transform")
        frame = self._to_frame(features, fitting=False)
        parts: list[np.ndarray] = []
        for column in self.columns_:
            values = frame[column.name]
            if column.kind == "numeric":
                median, mean, scale = self.numeric_stats_[column.name]
                numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
                numeric[~np.isfinite(numeric)] = np.nan
                filled = np.where(np.isfinite(numeric), numeric, median)
                parts.append(((filled - mean) / scale).astype(np.float32)[:, None])
            else:
                mapping = self.category_maps_[column.name]
                indexes = np.asarray([mapping.get(self._category_key(v), 0) for v in values.to_numpy()], dtype=np.int64)
                one_hot = np.zeros((len(indexes), len(mapping)), dtype=np.float32)
                one_hot[np.arange(len(indexes)), indexes] = 1.0
                parts.append(one_hot)
        if not parts:
            raise ValueError("preprocessor has no feature columns")
        return np.concatenate(parts, axis=1)

    def fit_transform(self, features) -> np.ndarray:
        return self.fit(features).transform(features)

    @property
    def output_dim(self) -> int:
        if self.columns_ is None:
            raise RuntimeError("fit must be called before output_dim")
        return len(self.output_names_)

    def state_dict(self) -> dict[str, object]:
        if self.columns_ is None:
            raise RuntimeError("cannot serialize an unfitted preprocessor")
        return {
            "columns": [(column.name, column.kind) for column in self.columns_],
            "numeric_stats": self.numeric_stats_,
            "category_maps": self.category_maps_,
            "output_names": self.output_names_,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.columns_ = tuple(FeatureColumn(name, kind) for name, kind in state["columns"])  # type: ignore[misc]
        self.numeric_stats_ = dict(state["numeric_stats"])  # type: ignore[arg-type]
        self.category_maps_ = dict(state["category_maps"])  # type: ignore[arg-type]
        self.output_names_ = tuple(state["output_names"])  # type: ignore[arg-type]
