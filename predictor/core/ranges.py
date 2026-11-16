"""A range around every number: where the real outcome will probably be.

A model always gives one number. A grower who plans on it needs to know how far off it can be. The trainer
measures that on rows the model did not learn from (conformal prediction, Vovk et al. 2005; Lei et al. 2018):
the distance that covered 80% of those rows becomes the half-width of the range. Where the training data
allows it there is a half-width per certainty level, because a prediction with few comparable cases in the
history is off by more than one with many.

The range is for the number itself (gas use, harvest). The difference between two scenarios has its own
margin: both sides share most of their error, so it is not the sum of two ranges.
"""

from __future__ import annotations

import math
from typing import Any

LEVEL = 0.8
MIN_ROWS_PER_LEVEL = 30
_ORDER = ["low", "medium", "high"]   # from least to most certain


def conformal_half_width(absolute_errors: list[float], level: float = LEVEL) -> float | None:
    """The error that `level` of unseen rows stay within (with the finite-sample correction)."""
    errors = sorted(e for e in absolute_errors if e == e)
    n = len(errors)
    if n < 10:
        return None
    rank = min(n, math.ceil((n + 1) * level))
    return float(errors[rank - 1])


def build(absolute_errors: list[float], certainty_levels: list[str] | None, level: float = LEVEL) -> dict[str, Any] | None:
    """The stored range: one half-width, plus one per certainty level that has enough rows."""
    half_width = conformal_half_width(absolute_errors, level)
    if half_width is None:
        return None
    spec: dict[str, Any] = {"level": level, "half_width": half_width, "calibration_rows": len(absolute_errors)}
    if certainty_levels is not None:
        by_level = {}
        for name in ("low", "medium", "high"):
            errors = [e for e, lv in zip(absolute_errors, certainty_levels) if lv == name]
            if len(errors) >= MIN_ROWS_PER_LEVEL:
                by_level[name] = conformal_half_width(errors, level)
        # Less certain is never narrower: with few rows a level's own number can come out too small by chance.
        widest = 0.0
        for name in ("high", "medium", "low"):
            if name in by_level:
                widest = by_level[name] = max(by_level[name], widest)
        if len(by_level) > 1:
            spec["by_certainty"] = by_level
    return spec


def half_width(spec: dict[str, Any] | None, certainty_level: str | None = None) -> float | None:
    if not spec:
        return None
    by_level = spec.get("by_certainty", {})
    if certainty_level in by_level or certainty_level not in _ORDER:
        return by_level.get(certainty_level, spec["half_width"])
    # A level without enough rows of its own: the overall width, but never narrower than a more certain level.
    more_certain = [by_level[name] for name in _ORDER[_ORDER.index(certainty_level) + 1:] if name in by_level]
    return max([spec["half_width"], *more_certain])


def around(spec: dict[str, Any] | None, value: float, certainty_level: str | None = None,
           bounds: list[float | None] | None = None) -> dict[str, Any] | None:
    """`{"low": .., "high": .., "level": 0.8}` for a predicted value, kept within the target's physical limits."""
    width = half_width(spec, certainty_level)
    if width is None:
        return None
    low, high = value - width, value + width
    if bounds:
        low = max(low, bounds[0]) if bounds[0] is not None else low
        high = min(high, bounds[1]) if bounds[1] is not None else high
    return {"low": round(low, 6), "high": round(high, 6), "level": spec["level"]}
