"""How well does the training data cover a situation? ("comparable cases")

A model can always produce a number, also for a situation it has hardly seen. For a
scenario that matters: the error of a predicted effect is about three times smaller
when there are 10+ comparable cases in the training data than when there are 0-2
(checked against the nursery simulator's answer key, see docs/scenarios.md).

Every input is standardised and weighted by how much the model relies on it, so a
difference in an input the model ignores does not make two cases incomparable. Two
cases are comparable when their distance is within a radius chosen so that a typical
training case has about ten comparable cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NEIGHBOURS = 10            # a typical training case has this many comparable cases
HIGH, MEDIUM = 10, 3       # certainty levels by number of comparable cases
MIN_WEIGHT = 0.02          # every input counts a little, also when the model barely uses it
CATEGORY_WEIGHT = 0.7      # a different category counts like 0.7 standard deviations
MAX_POINTS = 5000          # larger training sets are subsampled (and counts scaled up)
_CHUNK = 512


def build_support(matrix: np.ndarray, kinds: list[str], names: list[str], categories: list[list[str] | None],
                  importance: dict[str, float] | None, seed: int = 42) -> dict[str, Any]:
    """Summarise the training rows (float matrix, categories as codes) for later look-ups."""
    matrix = np.asarray(matrix, dtype=np.float64)
    importance = importance or {}
    weights = np.array([max(float(importance.get(n, 0.0)), MIN_WEIGHT) for n in names])
    weights = weights / weights.sum()
    features = []
    for i, (name, kind) in enumerate(zip(names, kinds)):
        column = matrix[:, i]
        if kind == "categorical":
            features.append({"name": name, "kind": kind, "weight": float(weights[i]),
                             "n_categories": len(categories[i] or [])})
        else:
            mean = float(np.nanmean(column)) if np.isfinite(column).any() else 0.0
            std = float(np.nanstd(column)) or 1.0
            features.append({"name": name, "kind": kind, "weight": float(weights[i]), "mean": mean, "std": std})
    points = _embed(features, matrix)
    scale = 1.0
    rng = np.random.default_rng(seed)
    if len(points) > MAX_POINTS:
        scale = len(points) / MAX_POINTS
        points = points[rng.choice(len(points), MAX_POINTS, replace=False)]
    sample = points[rng.choice(len(points), min(500, len(points)), replace=False)]
    distances = np.sort(_distances(sample, points), axis=1)
    # In a subsample, the 10th neighbour of the full data is roughly the (10 / scale)-th one.
    k = min(max(1, round(NEIGHBOURS / scale)), len(points) - 1)
    radius = float(np.median(distances[:, k])) if k > 0 else 0.0  # column 0 is the point itself
    return {"version": 1, "features": features, "radius": radius, "scale": scale,
            "rows": int(len(matrix)), "points": np.round(points, 4).tolist()}


class SupportIndex:
    def __init__(self, data: dict[str, Any]) -> None:
        self.features = data["features"]
        self.points = np.asarray(data["points"], dtype=np.float64)
        self.radius = float(data["radius"])
        self.scale = float(data.get("scale", 1.0))

    def comparable(self, matrix: np.ndarray) -> list[int]:
        """Number of comparable training cases per row (rows as the model's input matrix)."""
        queries = _embed(self.features, np.asarray(matrix, dtype=np.float64))
        counts = (_distances(queries, self.points) <= self.radius + 1e-9).sum(axis=1) * self.scale
        return [int(round(c)) for c in counts]


def certainty(count: int, case_label: str = "cases") -> dict[str, Any]:
    case_label = case_label if count != 1 else _singular(case_label)
    if count >= HIGH:
        level, message = "high", f"Based on {count} comparable {case_label} in the history."
    elif count >= MEDIUM:
        level, message = "medium", (f"Based on only {count} comparable {case_label} in the history; "
                                    "treat the size of the effect with some care.")
    else:
        found = f"only {count}" if count else "no"
        level, message = "low", (f"Uncertain: {found} comparable {case_label} in the history, "
                                 "so this is mostly extrapolation.")
    return {"comparable_cases": count, "level": level, "message": message}


def _singular(label: str) -> str:
    if label.endswith(("ches", "shes", "sses", "xes")):
        return label[:-2]   # batches -> batch
    return label[:-1] if label.endswith("s") else label   # fields -> field


def _embed(features: list[dict[str, Any]], matrix: np.ndarray) -> np.ndarray:
    parts = []
    for i, f in enumerate(features):
        column = matrix[:, i]
        w = np.sqrt(f["weight"])
        if f["kind"] == "categorical":
            onehot = np.zeros((len(column), max(f["n_categories"], 1)))
            valid = np.isfinite(column) & (column >= 0) & (column < f["n_categories"])
            onehot[np.nonzero(valid)[0], column[valid].astype(int)] = 1.0
            parts.append(onehot * CATEGORY_WEIGHT * w)
        else:
            z = (column - f["mean"]) / f["std"]
            parts.append(np.nan_to_num(z, nan=0.0)[:, None] * w)  # missing = average
    return np.hstack(parts) if parts else np.zeros((len(matrix), 0))


def _distances(queries: np.ndarray, points: np.ndarray) -> np.ndarray:
    out = np.empty((len(queries), len(points)))
    p2 = (points ** 2).sum(axis=1)
    for start in range(0, len(queries), _CHUNK):
        q = queries[start:start + _CHUNK]
        d2 = (q ** 2).sum(axis=1)[:, None] + p2[None, :] - 2 * q @ points.T
        out[start:start + _CHUNK] = np.sqrt(np.maximum(d2, 0.0))
    return out
