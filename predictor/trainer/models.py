"""Candidate models per task and their evaluation (sub-question 4).

Every use case gets a simple baseline and LightGBM; the better one on a
validation split wins. Anomaly use cases get an Isolation Forest. Preprocessing
lives inside the sklearn Pipeline, so it is exported together with the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

PRIMARY_METRIC = {"regression": ("mae", "lower"), "classification": ("f1", "higher"), "anomaly": (None, None)}


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split(n: int, test_fraction: float, chronological: bool, stratify: pd.Series | None) -> Split:
    """train / validation / test. Time-ordered data is split chronologically to avoid leakage."""
    index = np.arange(n)
    if chronological:
        n_test = int(round(n * test_fraction))
        n_val = int(round((n - n_test) * 0.2))
        return Split(index[: n - n_test - n_val], index[n - n_test - n_val : n - n_test], index[n - n_test :])
    strat = stratify if stratify is not None and stratify.value_counts().min() >= 5 else None
    rest, test = train_test_split(index, test_size=test_fraction, random_state=SEED, stratify=strat)
    strat_rest = strat.iloc[rest] if strat is not None else None
    train, val = train_test_split(rest, test_size=0.2, random_state=SEED, stratify=strat_rest)
    return Split(np.sort(train), np.sort(val), np.sort(test))


def candidates(task: str, anomaly_share: float = 0.05) -> dict[str, Pipeline]:
    impute = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    lgbm = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)
    if task == "regression":
        return {
            "linear_regression": Pipeline([impute, ("scale", StandardScaler()), ("model", LinearRegression())]),
            "lightgbm": Pipeline([impute, ("model", LGBMRegressor(**lgbm))]),
        }
    if task == "classification":
        return {
            "logistic_regression": Pipeline(
                [impute, ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
            ),
            "lightgbm": Pipeline([impute, ("model", LGBMClassifier(**lgbm))]),
        }
    if task == "anomaly":
        return {
            "isolation_forest": Pipeline(
                [impute, ("model", IsolationForest(n_estimators=200, contamination=anomaly_share, random_state=SEED))]
            )
        }
    raise ValueError(task)


def evaluate(task: str, model: Pipeline, X: pd.DataFrame, y: pd.Series | None,
             evaluation_label: pd.Series | None = None) -> dict[str, float]:
    X = X.to_numpy(dtype=np.float32)
    if task == "regression":
        pred = model.predict(X)
        return {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(root_mean_squared_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }
    if task == "classification":
        pred = model.predict(X)
        proba = model.predict_proba(X)
        binary = proba.shape[1] == 2
        metrics = {
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, average="binary" if binary else "macro", zero_division=0)),
        }
        if binary and y.nunique() == 2:
            metrics["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
        return metrics
    flagged = model.predict(X) == -1
    metrics = {"anomaly_rate": float(flagged.mean())}
    if evaluation_label is not None:
        truth = _truthy(evaluation_label)
        metrics.update(
            {
                "label_rate": float(truth.mean()),
                "precision_vs_label": float(precision_score(truth, flagged, zero_division=0)),
                "recall_vs_label": float(recall_score(truth, flagged, zero_division=0)),
            }
        )
    return metrics


def is_better(task: str, a: dict[str, float], b: dict[str, float] | None) -> bool:
    if b is None:
        return True
    metric, direction = PRIMARY_METRIC[task]
    return a[metric] < b[metric] if direction == "lower" else a[metric] > b[metric]


def feature_importance(model: Pipeline, names: list[str]) -> dict[str, float] | None:
    est = model.named_steps["model"]
    if isinstance(est, (LGBMRegressor, LGBMClassifier)):
        gain = est.booster_.feature_importance(importance_type="gain")
    elif isinstance(est, (LinearRegression, LogisticRegression)):
        gain = np.abs(np.atleast_2d(est.coef_)).sum(axis=0)
    else:
        return None
    total = float(np.sum(gain)) or 1.0
    ranked = sorted(zip(names, gain), key=lambda t: -t[1])
    return {name: round(float(g) / total, 4) for name, g in ranked}


def _truthy(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.fillna(0).astype(float).to_numpy() != 0
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y", "failure", "anomaly"}).to_numpy()


def describe_algorithm(name: str) -> str:
    return {
        "linear_regression": "Linear regression (baseline)",
        "logistic_regression": "Logistic regression (baseline)",
        "lightgbm": "LightGBM gradient-boosted trees",
        "isolation_forest": "Isolation Forest (unsupervised anomaly detection)",
    }.get(name, name)
