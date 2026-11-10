"""What a client (e.g. the dashboard) needs to know about the joins of a model, built from the model itself.

Nothing here knows a use case: titles, sentences, key lists and date ranges come from the join configuration and
the lookup snapshot, so every dataset with ``joins`` in its manifest is described the same way.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, Mapping

from predictor.core.joins import JoinEngine
from predictor.core.manifest import JoinBound, JoinConfig

MAX_KEY_CHOICES = 200  # more than this is a free-text input, not a dropdown
_VERBS = {"mean": "average", "sum": "total", "min": "lowest", "max": "highest", "count": "number of days with",
          "first": "first", "last": "last"}


def pretty(name: str) -> str:
    text = name.replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:]


def _edge(join: JoinConfig, bound: JoinBound) -> str:
    text = pretty(bound.column or join.time.base or "")
    unit = bound.unit[:-1] if abs(bound.offset) == 1 else bound.unit  # "1 day", "2 weeks"
    return f"{text} {'+' if bound.offset > 0 else '-'} {abs(bound.offset)} {unit}" if bound.offset else text


def _summary(join: JoinConfig) -> str:
    sources = ", ".join(dict.fromkeys(a.column for a in join.aggregate.values()))
    if join.time is None:
        return f"Takes {sources} from a table, matched on {', '.join(pretty(k) for k in join.keys)}."
    ops = " and ".join(dict.fromkeys(_VERBS[a.op] for a in join.aggregate.values()))
    per = f", per {', '.join(pretty(k) for k in join.keys)}" if join.keys else ""
    return f"{ops.capitalize()} of {sources} from {_edge(join, join.window.start)} up to {_edge(join, join.window.end)}{per}."


def describe_joins(engine: JoinEngine | None, descriptor: Mapping[str, Any], features: set[str],
                   units: Mapping[str, str], max_choices: int = MAX_KEY_CHOICES) -> list[dict[str, Any]]:
    if engine is None:
        return []
    trained = {j["name"]: j for j in descriptor.get("joins", [])}
    examples = descriptor.get("joins_example") or {}
    described = []
    for join in engine.joins:
        lookup = engine.lookups[join.name]
        info = trained.get(join.name, {})
        date_range = ([str(lookup.dates.min()), str(lookup.dates.max())]
                      if lookup.dates is not None and len(lookup.dates) else None)
        inputs = []
        for column in join.base_inputs:
            if column in join.keys:
                keys = lookup.known_values(list(join.keys).index(column))
                inputs.append({"name": column, "role": "key", "range": None, "example": examples.get(column),
                               "allowed_values": keys if len(keys) <= max_choices else None})
            else:
                inputs.append({"name": column, "role": "date", "range": date_range, "allowed_values": None,
                               "example": examples.get(column)})
        fills = [c for c in join.new_columns if c in features]
        described.append({
            "name": join.name, "title": pretty(join.name), "summary": _summary(join), "accepts": join.base_inputs,
            "fills": fills, "units": {c: units[c] for c in fills if c in units}, "fallback": join.fallback,
            "source_file": PurePath(info["source"]).name if info.get("source") else None,
            "lookup_rows": info.get("lookup_rows", len(lookup)), "lookup_range": date_range,
            "matched": info.get("matched"), "dropped_rows": info.get("dropped_rows", 0), "inputs": inputs,
        })
    return described
