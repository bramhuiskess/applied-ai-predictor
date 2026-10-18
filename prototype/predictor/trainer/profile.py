"""Turn an arbitrary tabular dataset into model-ready features plus a schema.

This is what makes the system work with "any" tabular dataset: column types,
binary flags and categories are inferred, and the result is written down as a
JSON Schema so serving validates exactly what training saw (Breck et al., 2019).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from predictor.core.manifest import Policy, TrainingConfig
from predictor.trainer.errors import TrainingError

MAX_CATEGORIES = 50
MAX_CLASSES = 50


@dataclass
class FeatureSpec:
    name: str
    kind: str  # number | integer | binary | categorical
    categories: list[str] | None = None


@dataclass
class PreparedData:
    X: pd.DataFrame  # numeric, categorical columns encoded as codes
    y: pd.Series | None  # encoded target (class index for classification)
    features: list[FeatureSpec]
    task: str
    target_name: str | None
    classes: list[str] | None
    evaluation_label: pd.Series | None
    persistence: pd.Series | None  # current value of the target, for horizon tasks
    chronological: bool
    notes: list[str] = field(default_factory=list)


def prepare(df: pd.DataFrame, cfg: TrainingConfig, policy: Policy) -> PreparedData:
    notes: list[str] = []
    df = df.rename(columns=cfg.rename_columns).copy()

    forbidden = {f.lower() for f in policy.forbidden_fields}
    clash = sorted(c for c in df.columns if str(c).lower() in forbidden)
    if clash:
        raise TrainingError(
            f"dataset contains forbidden column(s) {clash}; remove them from the export before training"
        )

    missing = [c for c in [*cfg.drop_columns, cfg.target, cfg.time_column, cfg.evaluation_label]
               if c and c not in df.columns]
    if missing:
        raise TrainingError(f"column(s) {missing} named in the manifest are not in the dataset {list(df.columns)}")
    df = df.drop(columns=cfg.drop_columns)

    chronological = False
    if cfg.time_column:
        df[cfg.time_column] = pd.to_datetime(df[cfg.time_column], errors="raise")
        df = df.sort_values(cfg.time_column, kind="stable").reset_index(drop=True)
        df = df.drop(columns=[cfg.time_column])
        chronological = True

    if policy.min_group_size:
        rule = policy.min_group_size
        too_small = df[rule.field] < rule.min
        if too_small.any():
            notes.append(f"dropped {int(too_small.sum())} rows with {rule.field} < {rule.min} (policy)")
            df = df[~too_small].reset_index(drop=True)

    task = cfg.task
    target_name = cfg.target
    persistence = None
    y_raw = None
    if cfg.target:
        if task == "auto":
            task = detect_task(df[cfg.target])
            notes.append(f"task detected automatically: {task}")
        if cfg.horizon_steps:
            if task != "regression":
                raise TrainingError("horizon_steps is only supported for regression")
            h = cfg.horizon_steps
            y_raw = df[cfg.target].shift(-h)
            persistence = df[cfg.target]
            target_name = f"{cfg.target} (+{h} steps)"
            notes.append(f"forecasting {cfg.target} {h} steps ahead; the current value stays a feature")
        else:
            y_raw = df[cfg.target]
            df = df.drop(columns=[cfg.target])
        keep = y_raw.notna()
        if not keep.all():
            notes.append(f"dropped {int((~keep).sum())} rows without a target value")
        df, y_raw = df[keep].reset_index(drop=True), y_raw[keep].reset_index(drop=True)
        if persistence is not None:
            persistence = persistence[keep].reset_index(drop=True)
    elif task == "auto":
        task = "anomaly"
        notes.append("no target given: training an anomaly detector")

    evaluation_label = None
    if cfg.evaluation_label:
        evaluation_label = df.pop(cfg.evaluation_label).reset_index(drop=True)

    specs, X = _profile_features(df, notes)
    if not specs:
        raise TrainingError("no usable feature columns left after profiling")

    classes = None
    y = None
    if y_raw is not None:
        if task == "classification":
            labels = y_raw.astype(str)
            classes = sorted(labels.unique())
            if len(classes) < 2:
                raise TrainingError("the target has only one class")
            if len(classes) > MAX_CLASSES:
                raise TrainingError(f"the target has {len(classes)} classes; at most {MAX_CLASSES} supported")
            y = labels.map({c: i for i, c in enumerate(classes)}).astype(int)
        else:
            if not pd.api.types.is_numeric_dtype(y_raw):
                raise TrainingError(f"regression target {cfg.target!r} must be numeric")
            y = y_raw.astype(float)

    if len(X) < 50:
        raise TrainingError(f"only {len(X)} usable rows; at least 50 are needed")

    return PreparedData(X, y, specs, task, target_name, classes, evaluation_label, persistence, chronological, notes)


def detect_task(target: pd.Series) -> str:
    values = target.dropna()
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return "classification"
    is_integer = np.allclose(values, np.round(values))
    return "classification" if is_integer and values.nunique() <= 10 else "regression"


def _profile_features(df: pd.DataFrame, notes: list[str]) -> tuple[list[FeatureSpec], pd.DataFrame]:
    specs: list[FeatureSpec] = []
    columns: dict[str, pd.Series] = {}
    for name in df.columns:
        col = df[name]
        non_null = col.dropna()
        if non_null.empty:
            notes.append(f"dropped column {name!r}: empty")
            continue
        if non_null.nunique() == 1:
            notes.append(f"dropped column {name!r}: constant")
            continue
        if pd.api.types.is_datetime64_any_dtype(col):
            notes.append(f"dropped column {name!r}: timestamps are not used as features")
            continue
        if pd.api.types.is_bool_dtype(col):
            specs.append(FeatureSpec(str(name), "binary"))
            columns[str(name)] = col.astype(float)
        elif pd.api.types.is_numeric_dtype(col):
            if set(non_null.unique()) <= {0, 1}:
                kind = "binary"
            elif np.allclose(non_null, np.round(non_null)):
                kind = "integer"
            else:
                kind = "number"
            specs.append(FeatureSpec(str(name), kind))
            columns[str(name)] = col.astype(float)
        else:
            values = non_null.astype(str)
            if values.nunique() > MAX_CATEGORIES:
                notes.append(f"dropped column {name!r}: {values.nunique()} distinct text values (identifier or free text?)")
                continue
            categories = sorted(values.unique())
            specs.append(FeatureSpec(str(name), "categorical", categories))
            codes = {c: i for i, c in enumerate(categories)}
            columns[str(name)] = col.map(lambda v: codes.get(str(v), np.nan) if pd.notna(v) else np.nan).astype(float)
    return specs, pd.DataFrame(columns)


def build_feature_schema(
    use_case_id: str, specs: list[FeatureSpec], X_train: pd.DataFrame, units: dict[str, str]
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for spec in specs:
        if spec.kind == "categorical":
            prop: dict[str, Any] = {"type": "string", "enum": spec.categories}
        elif spec.kind == "binary":
            prop = {"type": "integer", "enum": [0, 1]}
        else:
            # Whole-number columns (rpm, minutes) still accept decimals from sensors.
            prop = {"type": "number"}
        prop["x-kind"] = spec.kind
        if spec.kind in ("number", "integer"):
            column = X_train[spec.name].dropna()
            low, high = float(column.min()), float(column.max())
            prop["x-observed-range"] = [_tidy(low), _tidy(high)]
        if spec.name in units:
            prop["x-unit"] = units[spec.name]
        properties[spec.name] = prop
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:predictor:{use_case_id}:features",
        "title": f"Input features for {use_case_id}",
        "type": "object",
        "properties": properties,
        "required": [s.name for s in specs],
        "additionalProperties": False,
        "x-feature-order": [s.name for s in specs],
    }


def _tidy(value: float) -> float | int:
    return int(value) if float(value).is_integer() else round(value, 6)
