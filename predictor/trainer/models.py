"""Candidate models per task and their evaluation (sub-question 4).

Every use case gets a simple baseline and LightGBM (and for regression the
average of the two, the "blend"); the best one on a validation split wins. Anomaly use cases get an Isolation Forest. Preprocessing
lives inside the sklearn Pipeline, so it is exported together with the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import lsq_linear
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest, VotingRegressor
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

from predictor.trainer.errors import TrainingError

SEED = 42

PRIMARY_METRIC = {"regression": ("mae", "lower"), "classification": ("f1", "higher"), "anomaly": (None, None)}


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split(n: int, test_fraction: float, chronological: bool, stratify: pd.Series | None,
               test_start: int | None = None, test_mask: np.ndarray | None = None) -> Split:
    """train / validation / test. Time-ordered data is split chronologically to avoid leakage.

    `test_start` (rows are sorted by time) fixes where the test set begins, for backtests.
    """
    index = np.arange(n)
    if test_mask is not None:
        # Whole groups are the test set; the rest is split into train and validation.
        rest, test = index[~test_mask], index[test_mask]
        if chronological:
            n_val = int(round(len(rest) * 0.2))
            return Split(rest[: len(rest) - n_val], rest[len(rest) - n_val:], test)
        train, val = train_test_split(rest, test_size=0.2, random_state=SEED)
        return Split(np.sort(train), np.sort(val), test)
    if chronological:
        n_test = n - test_start if test_start is not None else int(round(n * test_fraction))
        n_val = int(round((n - n_test) * 0.2))
        return Split(index[: n - n_test - n_val], index[n - n_test - n_val : n - n_test], index[n - n_test :])
    strat = stratify if stratify is not None and stratify.value_counts().min() >= 5 else None
    rest, test = train_test_split(index, test_size=test_fraction, random_state=SEED, stratify=strat)
    strat_rest = strat.iloc[rest] if strat is not None else None
    train, val = train_test_split(rest, test_size=0.2, random_state=SEED, stratify=strat_rest)
    return Split(np.sort(train), np.sort(val), np.sort(test))


def indicator_layout(categories: list[int]) -> tuple[list[int], list[int], list[int]]:
    """(numeric features, the feature of each 0/1 column, the code of each 0/1 column) for `CategoryIndicators`.

    `categories` holds the number of categories per feature, 0 for numbers.
    """
    numeric = [i for i, k in enumerate(categories) if k < 2]
    columns = [i for i, k in enumerate(categories) if k >= 2 for _ in range(1, k)]
    codes = [code for k in categories if k >= 2 for code in range(1, k)]
    return numeric, columns, codes


class CategoryIndicators(BaseEstimator, TransformerMixin):
    """One 0/1 column per category, for the linear models.

    The profiler encodes a category as its place in the sorted values (AI4I's product quality: H=0, L=1, M=2).
    A tree can split on that, but a linear model reads it as a scale. The first category is the reference (the
    intercept carries it), so it gets no column; a code that matches no category (imputed) counts as the reference.
    The numbers come first, unchanged.
    """

    def __init__(self, categories: list[int] | None = None) -> None:
        self.categories = categories

    def fit(self, X, y=None):  # noqa: ARG002 - same signature as other transformers
        self.n_features_in_ = np.shape(X)[1]
        self.numeric_, self.columns_, self.codes_ = indicator_layout(self.categories or [0] * self.n_features_in_)
        return self

    def transform(self, X):
        X = np.asarray(X)
        indicators = (X[:, self.columns_] == np.asarray(self.codes_, dtype=X.dtype)).astype(X.dtype)
        return np.hstack([X[:, self.numeric_], indicators])

    def sources(self) -> list[int]:
        """The feature behind every column this step outputs."""
        return [*self.numeric_, *self.columns_]


class SignConstrainedLinearRegression(LinearRegression):
    """Least squares in which a declared effect fixes the sign of its coefficient.

    "More heating temperature = more gas" is physics; when the data (with correlated
    inputs) would give that coefficient the wrong sign, the coefficient becomes 0 instead
    and the other coefficients are fitted around it. Prediction and ONNX export are those
    of a normal linear regression. Inputs are standardised first, which keeps every sign.

    A tiny ridge penalty keeps the solution unique when inputs are redundant (lamp power and
    crop that always go together): without it the fit may return huge coefficients that cancel
    out, which float32 serving cannot reproduce.
    """

    RIDGE = 1e-4   # per row, on standardised inputs: negligible unless inputs are redundant

    def __init__(self, signs: list[int] | None = None) -> None:
        super().__init__()
        self.signs = signs

    def fit(self, X, y, sample_weight=None):  # noqa: ARG002 - same signature as LinearRegression
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        signs = np.asarray(self.signs if self.signs is not None else [0] * X.shape[1])
        lower = np.where(signs > 0, 0.0, -np.inf)
        upper = np.where(signs < 0, 0.0, np.inf)
        n, k = X.shape
        design = np.vstack([np.hstack([X, np.ones((n, 1))]),
                            np.hstack([np.sqrt(self.RIDGE * n) * np.eye(k), np.zeros((k, 1))])])  # intercept not penalised
        target = np.concatenate([y, np.zeros(k)])
        solution = lsq_linear(design, target, bounds=(np.append(lower, -np.inf), np.append(upper, np.inf)),
                              lsmr_tol="auto", method="bvls" if k < 200 else "trf").x
        self.coef_, self.intercept_ = solution[:-1], float(solution[-1])
        self.n_features_in_ = X.shape[1]
        return self


def _coefficients(pipeline: Pipeline, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """A linear pipeline's coefficients per feature: (the coefficient, the summed size of its coefficients).

    A category has a coefficient per 0/1 column and no direction, so its coefficient is 0 and its size the sum.
    """
    coef = np.atleast_2d(pipeline.named_steps["model"].coef_)
    step = pipeline.named_steps.get("indicators")
    sources = step.sources() if step else list(range(n_features))
    numeric = len(step.numeric_) if step else n_features
    signed, size = np.zeros(n_features), np.zeros(n_features)
    for column, feature in enumerate(sources):
        size[feature] += np.abs(coef[:, column]).sum()
        if column < numeric:
            signed[feature] = coef[0, column]
    return signed, size


def pinned_effects(model: Pipeline, names: list[str], monotone: list[int] | None) -> list[str]:
    """Declared effects the data did not support: the constrained linear model set them to 0."""
    est = model.named_steps.get("model")
    linear = est.named_estimators_["linear"] if isinstance(est, VotingRegressor) else model
    if not monotone or not isinstance(linear.named_steps.get("model"), SignConstrainedLinearRegression):
        return []
    coef, _ = _coefficients(linear, len(names))
    return [name for name, d, c in zip(names, monotone, coef) if d and abs(c) < 1e-9]


def candidates(task: str, anomaly_share: float = 0.05, monotone: list[int] | None = None,
               allowed: list[str] | None = None, categories: list[int] | None = None) -> dict[str, Pipeline]:
    """The models that may compete for a task; `allowed` (from the manifest) keeps only those names.

    `monotone` (one of -1/0/+1 per feature) becomes a LightGBM monotone constraint and,
    for regression, a sign constraint on the linear model's coefficients. `categories` (the number
    of categories per feature, 0 for numbers) gives the linear models a 0/1 column per category."""
    everything = _all_candidates(task, anomaly_share, monotone, categories)
    if allowed is None:
        return everything
    unknown = [name for name in allowed if name not in everything]
    if unknown:
        raise TrainingError(f"model(s) {unknown} are not available for a '{task}' output "
                            f"(available: {', '.join(everything)}); fix 'algorithms' in the manifest")
    return {name: model for name, model in everything.items() if name in allowed}


def _all_candidates(task: str, anomaly_share: float, monotone: list[int] | None,
                    categories: list[int] | None = None) -> dict[str, Pipeline]:
    impute = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    lgbm = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)
    if monotone and any(monotone):
        lgbm["monotone_constraints"] = monotone
    # Linear models get a 0/1 column per category; the trees keep the codes, they split on them fine.
    has_categories = bool(categories) and any(k >= 2 for k in categories)

    def indicators():
        return [("indicators", CategoryIndicators(categories))] if has_categories else []

    if task == "regression":
        signs = monotone
        if has_categories and monotone:   # the numbers come first, then the 0/1 columns (no declared effect)
            numeric, columns, _ = indicator_layout(categories)
            signs = [monotone[i] for i in numeric] + [0] * len(columns)

        def linear():
            return SignConstrainedLinearRegression(signs) if monotone and any(monotone) else LinearRegression()
        # The linear model extrapolates sensibly, the trees catch thresholds and interactions; on
        # small real datasets their average is often better than either (both obey the declared
        # effects, so the average does too).
        blend = VotingRegressor([("linear", Pipeline([*indicators(), ("scale", StandardScaler()), ("model", linear())])),
                                 ("lightgbm", LGBMRegressor(**lgbm))])
        return {
            "linear_regression": Pipeline([impute, *indicators(), ("scale", StandardScaler()), ("model", linear())]),
            "lightgbm": Pipeline([impute, ("model", LGBMRegressor(**lgbm))]),
            "blend": Pipeline([impute, ("model", blend)]),
        }
    if task == "classification":
        return {
            "logistic_regression": Pipeline(
                [impute, *indicators(), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
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


THRESHOLDS = [round(0.05 + 0.025 * i, 3) for i in range(37)]   # 0.05 .. 0.95
MIN_F1_GAIN = 0.02        # leave 0.5 only for a clearly better F1 ...
MIN_COST_SAVING = 0.05    # ... or a clearly lower cost


def predict_labels(model: Pipeline, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Class indices; with a yes/no `threshold`, "yes" (index 1) from that probability on."""
    if threshold is None:
        return model.predict(X)
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


def choose_threshold(y: pd.Series, probability: np.ndarray, costs: dict[str, float] | None = None) -> float:
    """The yes/no cutoff with the best F1 on rows the model did not learn from, or the lowest cost with `costs`.

    A cutoff of 0.5 misses many rare events (3% machine failures) when the model is unsure about them. But with
    sharp probabilities (LightGBM on AI4I) the F1 curve is flat between 0.4 and 0.6 and its best point is noise, so
    0.5 stays unless another cutoff is clearly better. Ties go to the cutoff closest to 0.5.
    """
    truth = np.asarray(y, dtype=int) == 1
    if truth.sum() < 5 or truth.all():
        return 0.5   # too few "yes" rows to choose from
    by_closeness = sorted(THRESHOLDS, key=lambda t: abs(t - 0.5))
    if costs:
        cost = {t: costs["missed"] * np.sum(truth & (probability < t))
                + costs["false_alarm"] * np.sum(~truth & (probability >= t)) for t in THRESHOLDS}
        best = min(by_closeness, key=cost.get)
        return best if cost[best] <= (1 - MIN_COST_SAVING) * cost[0.5] else 0.5
    f1 = {t: f1_score(truth, probability >= t, zero_division=0) for t in THRESHOLDS}
    best = max(by_closeness, key=f1.get)
    return best if f1[best] - f1[0.5] >= MIN_F1_GAIN else 0.5


def evaluate(task: str, model: Pipeline, X: pd.DataFrame, y: pd.Series | None,
             evaluation_label: pd.Series | None = None, bounds: tuple | None = None,
             threshold: float | None = None) -> dict[str, float]:
    """Metrics on a split. `bounds` clips regression predictions like the API does; `threshold` is the yes/no cutoff."""
    X = X.to_numpy(dtype=np.float32)
    if task == "regression":
        pred = model.predict(X)
        if bounds:
            pred = np.clip(pred, bounds[0] if bounds[0] is not None else -np.inf,
                           bounds[1] if bounds[1] is not None else np.inf)
        return {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(root_mean_squared_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }
    if task == "classification":
        pred = predict_labels(model, X, threshold)
        proba = model.predict_proba(X)
        binary = proba.shape[1] == 2
        metrics = {
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, average="binary" if binary else "macro", zero_division=0)),
        }
        if binary:   # the share of real "yes" cases it catches, and the share of its "yes" answers that is right
            metrics["recall"] = float(recall_score(y, pred, zero_division=0))
            metrics["precision"] = float(precision_score(y, pred, zero_division=0))
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


def effect_violations(model: Pipeline, names: list[str], monotone: list[int] | None) -> list[str]:
    """Declared directions a linear model contradicts (tree models get them as hard constraints)."""
    est = model.named_steps["model"]
    if not monotone or not isinstance(est, (LinearRegression, LogisticRegression)):
        return []
    coef, _ = _coefficients(model, len(names))  # scaled inputs: the sign is the same as for the raw input
    return [f"{name} should be {'increasing' if d > 0 else 'decreasing'} but the model says "
            f"{'increasing' if c > 0 else 'decreasing'}"
            for name, d, c in zip(names, monotone, coef) if d and c and np.sign(c) != d]


SIMPLICITY = ["linear_regression", "logistic_regression", "blend", "lightgbm"]


def one_standard_error_choice(task: str, validation: dict[str, dict[str, float]], best: str) -> str:
    """The simplest candidate whose cross-validated score is within one standard error of the best.

    A textbook rule (Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 7.10):
    with few groups the difference between candidates is often noise, and a simpler model
    extrapolates more predictably. For scenarios it matters doubly: a tree model can show a
    lever as having no effect at all because its prediction is flat between two splits.
    """
    metric, direction = PRIMARY_METRIC[task]
    if metric is None or f"{metric}_se" not in validation[best]:
        return best
    limit = validation[best][metric] + (1 if direction == "lower" else -1) * validation[best][f"{metric}_se"]
    for name in sorted(validation, key=lambda n: SIMPLICITY.index(n) if n in SIMPLICITY else len(SIMPLICITY)):
        score = validation[name][metric]
        if (score <= limit) if direction == "lower" else (score >= limit):
            return name
    return best


def is_better(task: str, a: dict[str, float], b: dict[str, float] | None) -> bool:
    if b is None:
        return True
    metric, direction = PRIMARY_METRIC[task]
    return a[metric] < b[metric] if direction == "lower" else a[metric] > b[metric]


def feature_importance(model: Pipeline, names: list[str]) -> dict[str, float] | None:
    est = model.named_steps["model"]
    if isinstance(est, VotingRegressor):   # the average of both parts' shares
        parts = [feature_importance(e if isinstance(e, Pipeline) else Pipeline([("model", e)]), names)
                 for e in est.estimators_]
        merged = {n: sum(p.get(n, 0.0) for p in parts if p) / len(parts) for n in names}
        return {n: round(v, 4) for n, v in sorted(merged.items(), key=lambda t: -t[1])}
    if isinstance(est, (LGBMRegressor, LGBMClassifier)):
        gain = est.booster_.feature_importance(importance_type="gain")
    elif isinstance(est, (LinearRegression, LogisticRegression)):
        _, gain = _coefficients(model, len(names))
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
        "blend": "Blend: a mix of the linear model and LightGBM",
        "isolation_forest": "Isolation Forest (unsupervised anomaly detection)",
    }.get(name, name)
