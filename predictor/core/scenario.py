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
                        "value_eur": config.value_eur if config else None})
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
