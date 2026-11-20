"""The best plan: try every setting of the levers and propose the one with the best result.

A scenario answers "what if I do this?". A grower's real question is "what should I do?". The finder lays a
grid over the levers (their own range and step from the manifest), calculates every plan with the same engine
as a scenario, and ranks them by the objective: the net result in euros, or one output to make as high or as
low as possible.

An optimiser pushed against a model walks to wherever the model is most optimistic, so the finder is careful
in two ways, both set on evidence (the tomato simulator knows the true result of every plan):

* **A gain has to be worth mentioning.** Promised and true gains agree well (correlation 0.96), but the finder
  promises about a quarter too much, and small promises are unreliable: of the advice that promised less than
  1 EUR/m2, about 4 in 10 made things worse; above 2 EUR/m2 almost none did. So a plan is only advised when its
  gain is at least `MIN_GAIN_SHARE` of the model's typical error in the same unit. Otherwise: keep the current plan.
* **Never outside the training data.** Plans with an input outside what the model was trained on are left out.

A limit on certainty (comparable cases) is available as `min_certainty`, but it is off by default: in the same
test it removed two thirds of the gain and did not make the advice more reliable.

With many levers the grid is coarse: 2,000 plans over the nursery's five levers leave four values per lever
(temperature 17, 19, 21 and 23 degrees). So the finder then refines the best plans: it tries each coarse lever
on its own steps between the grid values, keeps a better plan and repeats until nothing improves. In the
nursery simulator that gave 5% more true gain per batch (`scripts/check_scenarios.py --finder`).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable

from predictor.core.manifest import ScenarioConfig

MAX_PLANS = 2000
MIN_GAIN_SHARE = 0.4      # of the typical error; with it 7% of the advice made things worse instead of 16%
CERTAINTY_ORDER = {"low": 0, "medium": 1, "high": 2}
REFINE_STARTS = 3         # refine the best 3 plans of the grid
REFINE_ROUNDS = 4         # at most 4 passes over the coarse levers each


def lever_values(low: float, high: float, step: float, at_most: int) -> list[float]:
    """The lever's own grid from min to max, thinned evenly (ends kept) when there are more than `at_most` values."""
    count = int(round((high - low) / step)) + 1
    values = [round(low + i * step, 10) for i in range(count)]
    if count <= at_most:
        return values
    picks = sorted({round(i * (count - 1) / (at_most - 1)) for i in range(at_most)})
    return [values[i] for i in picks]


def axes(scenario: ScenarioConfig, levers: list[str], max_plans: int = MAX_PLANS) -> dict[str, list[float]]:
    """The values of every lever in the grid: its own steps, thinned evenly when there are too many plans."""
    per_lever = max(2, int(max_plans ** (1 / len(levers))))
    return {name: lever_values(scenario.levers[name].min, scenario.levers[name].max, scenario.levers[name].step, per_lever)
            for name in levers}


def grid(scenario: ScenarioConfig, levers: list[str], max_plans: int = MAX_PLANS) -> list[dict[str, float]]:
    """Every combination of lever values, at most `max_plans` (coarser per lever when needed)."""
    values = axes(scenario, levers, max_plans).values()
    return [dict(zip(levers, combination)) for combination in itertools.product(*values)]


def coarse_levers(scenario: ScenarioConfig, levers: list[str], max_plans: int = MAX_PLANS) -> dict[str, float]:
    """The levers the grid could not try on every step, with the largest gap between two values it tried."""
    coarse = {}
    for name, values in axes(scenario, levers, max_plans).items():
        lever = scenario.levers[name]
        if len(values) < round((lever.max - lever.min) / lever.step) + 1:
            coarse[name] = max(b - a for a, b in zip(values, values[1:]))
    return coarse


def neighbours(scenario: ScenarioConfig, plan: dict[str, float], lever: str, gap: float) -> list[dict[str, float]]:
    """The plan with `lever` on each of its own steps within `gap` of its value."""
    config, value = scenario.levers[lever], plan[lever]
    first = math.ceil((max(config.min, value - gap) - config.min) / config.step - 1e-9)
    last = math.floor((min(config.max, value + gap) - config.min) / config.step + 1e-9)
    values = [round(config.min + k * config.step, 10) for k in range(first, last + 1)]
    return [{**plan, lever: v} for v in values if abs(v - value) > 1e-9]


def refine(scenario: ScenarioConfig, candidates: list[dict[str, Any]], coarse: dict[str, float],
           calculate: Callable[[list[dict[str, float]]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Look between the grid values around the best plans (see the module doc); returns the plans it added.

    A candidate is a dict with at least `plan`, `score` and `ok` (may be advised); `calculate` turns plans into them.
    """
    known = {_key(c["plan"]): c for c in candidates}
    added = []
    for current in sorted((c for c in candidates if c["ok"]), key=lambda c: -c["score"])[:REFINE_STARTS]:
        for _ in range(REFINE_ROUNDS):
            moved = False
            for lever, gap in coarse.items():
                nearby = neighbours(scenario, current["plan"], lever, gap)
                new = [p for p in nearby if _key(p) not in known]
                for c in calculate(new) if new else []:
                    known[_key(c["plan"])] = c
                    added.append(c)
                options = [known[_key(p)] for p in nearby]
                best = max((c for c in options if c["ok"]), key=lambda c: c["score"], default=current)
                if best["score"] > current["score"]:
                    current, moved = best, True
            if not moved:
                break
    return added


def _key(plan: dict[str, float]) -> tuple:
    return tuple(sorted(plan.items()))


def allowed(certainty_level: str | None, minimum: str) -> bool:
    """Without a map of the training data (no certainty) every plan counts; otherwise at least `minimum`."""
    return certainty_level is None or CERTAINTY_ORDER[certainty_level] >= CERTAINTY_ORDER[minimum]


def describe_objective(objective: dict[str, Any] | None) -> str:
    if objective is None:
        return "the highest net result in euros"
    return f"the {objective['goal']} {objective['output']}"


def typical_error_of_objective(margins: dict[str, float | None], prices: dict[str, float | None],
                               objective: dict[str, Any] | None) -> float | None:
    """The model's typical error in the unit of the objective: euros for the net result, else the output's own unit.

    For the net result the errors of the priced outputs are combined as independent errors (root of the sum of squares).
    """
    if objective is not None:
        return margins.get(objective["output"])
    parts = [(margins[name] * abs(price)) ** 2 for name, price in prices.items() if price is not None and margins.get(name) is not None]
    return sum(parts) ** 0.5 if parts else None
