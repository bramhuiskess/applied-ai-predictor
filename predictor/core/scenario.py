"""Scenario variables ("levers"): which inputs a user may change, within which range.

The manifest declares them; this module checks them against the trained model's
feature schema and describes them for clients (sliders in a dashboard, the
scenario input of the API).
"""

from __future__ import annotations

from typing import Any

from predictor.core.manifest import ManifestError, ScenarioConfig

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
