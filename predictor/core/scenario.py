"""Scenario variables ("levers"): which inputs a user may change, within which range.

The manifest declares them; this module checks them against the trained model's
feature schema and describes them for clients (sliders in a dashboard, the
scenario input of the API).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from predictor.core.manifest import ManifestError, ScenarioConfig
from predictor.core.validation import FieldError

LEVER_KINDS = {"number", "integer", "binary"}


def check_scenario(use_case_id: str, scenario: ScenarioConfig | None, feature_schema: dict[str, Any]) -> None:
    """A lever must be a numeric input of the trained model; fail at startup if not."""
    if scenario is None:
        return
    properties = feature_schema["properties"]
    for name, lever in scenario.levers.items():
        prop = properties.get(name)
        if prop is None:
            raise ManifestError(f"{use_case_id}: scenario lever {name!r} is not an input of the trained model "
                                f"(inputs: {', '.join(feature_schema['x-feature-order'])}); retrain or fix the manifest")
        if prop.get("x-kind") not in LEVER_KINDS:
            raise ManifestError(f"{use_case_id}: scenario lever {name!r} is {prop.get('x-kind')}; "
                                "only numeric and yes/no inputs can be levers")
        if prop.get("x-kind") == "binary" and (lever.min, lever.max, lever.step) != (0, 1, 1):
            raise ManifestError(f"{use_case_id}: scenario lever {name!r} is yes/no, so use min 0, max 1, step 1")
    for name, output in scenario.outputs.items():
        if output.price_from and properties.get(output.price_from, {}).get("x-kind") not in ("number", "integer"):
            raise ManifestError(f"{use_case_id}: scenario output {name!r} takes its price from {output.price_from!r}, "
                                "which is not a numeric input of the trained model")


def describe_scenario(scenario: ScenarioConfig | None, feature_schema: dict[str, Any],
                      descriptor: dict[str, Any]) -> dict[str, Any] | None:
    """The scenario section of GET /models/{id}: levers with the range the model has seen."""
    if scenario is None:
        return None
    properties = feature_schema["properties"]
    levers = []
    for name, lever in scenario.levers.items():
        prop = properties[name]
        seen = prop.get("x-observed-range")
        levers.append({
            "name": name,
            "label": lever.label or name,
            "unit": lever.unit or prop.get("x-unit"),
            "kind": prop.get("x-kind"),
            "min": lever.min,
            "max": lever.max,
            "step": lever.step,
            "effects": lever.effects,
            "training_range": seen,
            # Outside the data the model learned from, tree models predict a flat line.
            "beyond_training_data": bool(seen) and (lever.min < seen[0] or lever.max > seen[1]),
        })
    outputs = []
    for name in [descriptor.get("target_name")] if descriptor.get("target_name") else []:
        config = scenario.outputs.get(name)
        outputs.append({"name": name, "label": (config.label if config and config.label else name),
                        "unit": descriptor.get("target_unit"),
                        "value_eur": config.value_eur if config else None,
                        "price_from": config.price_from if config else None,
                        "is_cost": config.is_cost if config else False,
                        "margin": margin_of_error(descriptor)})
    return {"levers": levers, "outputs": outputs,
            "constraints": descriptor.get("scenario_constraints")}


# --------------------------------------------------------------------------- scenario input (ticket 3)

MAX_SCENARIOS = 20
_EPSILON = 1e-9

Validate = Callable[[dict[str, Any]], tuple[list[FieldError], list[str]]]


@dataclass
class ResolvedScenario:
    name: str
    features: dict[str, Any]
    changes: list[dict[str, Any]]
    warnings: list[str]


@dataclass
class ResolvedRequest:
    base: dict[str, Any]
    base_warnings: list[str]
    scenarios: list[ResolvedScenario]


def resolve_scenarios(
    scenario: ScenarioConfig,
    feature_schema: dict[str, Any],
    base: dict[str, Any],
    requested: list[dict[str, Any]],
    validate: Validate,
) -> tuple[ResolvedRequest | None, list[dict[str, Any]]]:
    """Turn "base + changes" into complete, validated input rows, one per scenario.

    `requested` holds {"name": str, "changes": {lever: {"set": x} | {"delta": d}}}. `validate`
    applies the use case's policy and feature schema to one row. Returns (resolved, errors);
    errors carry the scenario name (None for the base) so a client can point at the problem.
    """
    errors: list[dict[str, Any]] = []
    base_errors, base_warnings = validate(base)
    if base_errors:
        return None, [{"scenario": None, "field": e.field, "message": e.message} for e in base_errors]

    properties = feature_schema["properties"]
    seen_names: set[str] = set()
    resolved: list[ResolvedScenario] = []
    for item in requested:
        name = item["name"].strip()
        problems: list[tuple[str, str]] = []
        if not name:
            problems.append(("name", "a scenario needs a name"))
        elif name.lower() in seen_names:
            problems.append(("name", f"the name {name!r} is used twice; give every scenario its own name"))
        seen_names.add(name.lower())

        features = dict(base)
        changes = []
        for lever_name, change in item["changes"].items():
            lever = scenario.levers.get(lever_name)
            if lever is None:
                problems.append((lever_name, f"{lever_name!r} cannot be changed in a scenario; "
                                             f"you can change: {', '.join(scenario.levers)}"))
                continue
            before = base[lever_name]
            if "set" in change:
                after, how = change["set"], f"set to {_fmt(change['set'])}"
            else:
                after, how = before + change["delta"], f"{_fmt(before)} {change['delta']:+g}"
            after = round(after, 10)  # 20.1 + 0.2 is 20.3, not 20.300000000000001
            if not lever.min - _EPSILON <= after <= lever.max + _EPSILON:
                problems.append((lever_name, f"{lever_name} would become {_fmt(after)} ({how}), outside the "
                                             f"allowed range {_fmt(lever.min)} - {_fmt(lever.max)}"))
                continue
            if float(after).is_integer():
                after = int(after)  # 20 instead of 20.0; yes/no inputs stay 0 or 1
            features[lever_name] = after
            changes.append({"lever": lever_name, "label": lever.label or lever_name,
                            "unit": lever.unit or properties[lever_name].get("x-unit"),
                            "from": before, "to": features[lever_name]})

        if not problems:
            row_errors, row_warnings = validate(features)
            problems += [(e.field, e.message) for e in row_errors]
        if problems:
            errors += [{"scenario": name or None, "field": field, "message": message} for field, message in problems]
            continue
        warnings = [w for w in row_warnings if w not in base_warnings]
        if all(c["from"] == c["to"] for c in changes):
            warnings.append("this scenario changes nothing compared to the base")
        resolved.append(ResolvedScenario(name, features, changes, warnings))

    if errors:
        return None, errors
    return ResolvedRequest(base, base_warnings, resolved), []


def _fmt(value: float) -> str:
    return f"{value:g}"


# --------------------------------------------------------------------------- sweep (ticket 4)

MAX_SWEEP_POINTS = 201


def sweep_values(scenario: ScenarioConfig, lever_name: str, start: float | None, stop: float | None,
                 step: float | None) -> tuple[list[float], list[dict[str, Any]]]:
    """The values to try for one lever: from start to stop (default: the lever's whole range)."""
    lever = scenario.levers.get(lever_name)
    if lever is None:
        return [], [{"scenario": None, "field": "lever", "message": f"{lever_name!r} cannot be changed in a "
                     f"scenario; you can change: {', '.join(scenario.levers)}"}]
    start = lever.min if start is None else start
    stop = lever.max if stop is None else stop
    step = lever.step if step is None else step
    problems = []
    if not lever.min - _EPSILON <= start < stop <= lever.max + _EPSILON:
        problems.append(f"the sweep {_fmt(start)} - {_fmt(stop)} must go upwards and stay within the allowed "
                        f"range {_fmt(lever.min)} - {_fmt(lever.max)}")
    elif (stop - start) / step + 1 > MAX_SWEEP_POINTS + _EPSILON:
        problems.append(f"{_fmt(start)} - {_fmt(stop)} in steps of {_fmt(step)} gives more than "
                        f"{MAX_SWEEP_POINTS} points; use a bigger step")
    if problems:
        return [], [{"scenario": None, "field": "lever", "message": m} for m in problems]
    count = int((stop - start) / step + _EPSILON) + 1
    values = [round(start + i * step, 10) for i in range(count)]
    if stop - values[-1] > _EPSILON:
        values.append(stop)  # always include the end of the range
    return [int(v) if float(v).is_integer() else v for v in values], []


# --------------------------------------------------------------------------- difference base vs. scenario (ticket 5)

def margin_of_error(descriptor: dict[str, Any]) -> float | None:
    """The model's typical error on one prediction (test MAE), for regression."""
    if descriptor.get("task_type") != "regression":
        return None
    mae = (descriptor.get("metrics") or {}).get("mae")
    return round(float(mae), 6) if mae is not None else None


def compare(scenario: ScenarioConfig, descriptor: dict[str, Any], base_row: dict[str, Any],
            base_out: dict[str, Any], row: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """What changes from the base to a scenario, per output, in units, %, euros and against the margin."""
    result: dict[str, Any] = {}
    money: list[float] = []
    for name, now in out.items():
        before = base_out[name]
        config = scenario.outputs.get(name)
        if "probabilities" in now:
            result[name] = _compare_classes(before, now)
            continue
        change = round(now["value"] - before["value"], 6)
        item: dict[str, Any] = {
            "base": before["value"],
            "scenario": now["value"],
            "change": change,
            "change_pct": round(change / abs(before["value"]) * 100, 2) if before["value"] else None,
        }
        eur = _money(config, row, now["value"]) - _money(config, base_row, before["value"]) if _priced(config) else None
        item["eur"] = round(eur, 2) if eur is not None else None
        if eur is not None:
            money.append(eur)
        margin = margin_of_error(descriptor)
        item["margin"] = margin
        item["exceeds_margin"] = None if margin is None else abs(change) >= margin
        result[name] = item
    return {"outputs": result, "net_eur": round(sum(money), 2) if money else None}


def summarise(scenario: ScenarioConfig, descriptor: dict[str, Any], name: str, comparison: dict[str, Any]) -> str:
    """One plain sentence per scenario, for people who do not read JSON."""
    parts = []
    for output, item in comparison["outputs"].items():
        config = scenario.outputs.get(output)
        label = (config.label if config and config.label else output).lower()
        unit = descriptor.get("target_unit") or ""
        if "positive_class" in item:  # a yes/no or category output
            parts.append(f"the chance of {label} = {item['positive_class']} goes from "
                         f"{item['base_probability']:.0%} to {item['scenario_probability']:.0%}")
            continue
        if item["change"] == 0:
            parts.append(f"{label} does not change")
            continue
        direction = "higher" if item["change"] > 0 else "lower"
        size = f"{label} {abs(item['change_pct']):.1f}% {direction}" if item["change_pct"] is not None \
            else f"{label} {direction}"
        detail = f"{item['change']:+,.{_decimals(item['change'])}f} {unit}".strip()
        if item["eur"] is not None:  # positive = better for the grower (more income or lower cost)
            detail += f", worth {'+' if item['eur'] >= 0 else '-'}€{abs(item['eur']):,.0f}"
        sentence = f"{size} ({detail})"
        if item["exceeds_margin"] is False:
            sentence += (f"; this is within the model's typical error of ±{item['margin']:,.{_decimals(item['margin'])}f} "
                         f"{unit}, so treat it as no clear effect").rstrip()
        parts.append(sentence)
    text = f"{name}: " + "; ".join(parts)
    if comparison["net_eur"] is not None and len(comparison["outputs"]) > 1:
        text += f". Net worth {'+' if comparison['net_eur'] >= 0 else '-'}€{abs(comparison['net_eur']):,.0f}"
    return text + "."


def _priced(config) -> bool:
    return config is not None and (config.value_eur is not None or bool(config.price_from))


def _money(config, row: dict[str, Any], value: float) -> float:
    if config.value_eur is not None:
        return value * config.value_eur
    price = float(row[config.price_from])
    return value * price * (-1 if config.is_cost else 1)


def _compare_classes(before: dict[str, Any], now: dict[str, Any]) -> dict[str, Any]:
    classes = list(before["probabilities"])
    positive = classes[-1]  # for yes/no outcomes the "yes" class sorts last ("1", "true", "yes")
    return {
        "base": before["value"],
        "scenario": now["value"],
        "label_changed": before["value"] != now["value"],
        "positive_class": positive,
        "base_probability": before["probabilities"][positive],
        "scenario_probability": now["probabilities"][positive],
        "probability_change": {c: round(now["probabilities"][c] - before["probabilities"][c], 6) for c in classes},
    }


def _decimals(value: float) -> int:
    return 0 if abs(value) >= 100 else 1 if abs(value) >= 1 else 2
