"""The prediction service: identical for every use case.

Flow per request: policy check -> feature-schema validation -> conversion to a
typed matrix -> model runner -> normalised JSON result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from predictor.core import plan as planning
from predictor.core import ranges
from predictor.core.certainty import SupportIndex, certainty
from predictor.core.join_info import describe_joins
from predictor.core.manifest import Manifest, ManifestError
from predictor.core.monitoring import Monitor
from predictor.core.policy import check_record, check_schema
from predictor.core.scenario import (
    ResolvedRequest,
    check_scenario,
    compare,
    describe_scenario,
    margin_of_error,
    resolve_scenarios,
    summarise,
    sweep_values,
)
from predictor.core.ports import LoadedModel, ModelRegistry
from predictor.core.validation import FeatureValidator, FieldError

SCHEMA_VERSION = "1.0"
STREAM_CHUNK_SIZE = 500


class UseCaseNotFound(Exception):
    pass


class UseCaseNotReady(Exception):
    pass


class RequestRejected(Exception):
    def __init__(self, errors: list[dict[str, Any]], detail: str = "invalid records") -> None:
        super().__init__(f"{len(errors)} invalid record(s)")
        self.errors = errors
        self.detail = detail


@dataclass
class UseCase:
    manifest: Manifest
    model: LoadedModel | None
    validator: FeatureValidator | None
    supports: dict[str, SupportIndex] | None = None  # comparable-cases map per output (target name)
    alias_of: tuple[str, str] | None = None  # (use case, output) when this is a single-output view of another one

    @property
    def status(self) -> str:
        return "ready" if self.model else "not_trained"

    @property
    def descriptors(self) -> dict[str, dict[str, Any]]:
        """target name -> descriptor of every output (what the scenario code needs for margins and units)."""
        return {om.descriptor.get("target_name") or om.name: om.descriptor for om in self.model.outputs.values()}


@dataclass
class UnparsableRecord:
    """Placeholder for a JSONL line that could not be parsed."""

    message: str


@dataclass
class _Checked:
    record_id: str
    features: Any
    errors: list[FieldError]
    warnings: list[str]
    filled: dict[str, Any] | None = None  # columns the joins filled in (shown to the caller, absent without joins)
    assumed: list[str] | None = None      # unknown inputs (weather ahead) the caller left out: set to their usual value


class PredictionService:
    def __init__(self, use_cases: list[UseCase], monitor: Monitor | None = None) -> None:
        self._use_cases = {uc.manifest.use_case_id: uc for uc in use_cases}
        self.monitor = monitor or Monitor()

    @classmethod
    def from_manifests(
        cls, manifests: list[Manifest], registry: ModelRegistry, monitor: Monitor | None = None
    ) -> PredictionService:
        use_cases = []
        for manifest in manifests:
            model = registry.resolve(manifest.use_case_id, manifest.model.version)
            validator = None
            if model is not None:
                check_schema(manifest.use_case_id, manifest.policy, model.feature_schema)
                check_scenario(manifest.use_case_id, manifest.scenario, model.feature_schema)
                _check_joined_levers(manifest, model)
                _check_outputs_match(manifest, model)
                validator = FeatureValidator(
                    model.feature_schema, reject_unknown_fields=manifest.policy.reject_unknown_fields
                )
            supports = _supports(manifest, model)
            use_cases.append(UseCase(manifest, model, validator, supports))
            for alias, output in manifest.aliases.items():  # an old id keeps working as one output of this use case
                view = manifest.single_output(output, alias)
                only = model.only(output) if model else None
                use_cases.append(UseCase(view, only, validator, _supports(view, only), (manifest.use_case_id, output)))
        return cls(use_cases, monitor)

    # ------------------------------------------------------------------ queries

    def list_use_cases(self) -> list[dict[str, Any]]:
        return [
            {
                "use_case_id": uc.manifest.use_case_id,
                "description": uc.manifest.description,
                "status": uc.status,
                "task_type": uc.model.descriptor["task_type"] if uc.model else None,
                "model_version": uc.model.version if uc.model else None,
                "outputs": list(uc.model.outputs) if uc.model else None,
                "alias_of": {"use_case_id": uc.alias_of[0], "output": uc.alias_of[1]} if uc.alias_of else None,
            }
            for uc in self._use_cases.values()
        ]

    def describe(self, use_case_id: str) -> dict[str, Any]:
        uc = self._ready(use_case_id)
        descriptor = uc.model.descriptor
        schema = uc.model.feature_schema
        return {
            "use_case_id": use_case_id,
            "description": uc.manifest.description,
            "model_version": uc.model.version,
            "task_type": descriptor["task_type"],
            "algorithm": descriptor["algorithm"],
            "runner": descriptor["runner"],
            "target_name": descriptor.get("target_name"),
            "target_unit": descriptor.get("target_unit"),
            "classes": descriptor.get("classes"),
            "metrics": descriptor.get("metrics"),
            "trained_at": descriptor.get("trained_at"),
            "outputs": {name: _summary(om.descriptor, "classes", "metrics", "trained_at")
                        for name, om in uc.model.outputs.items()},
            "inputs": [
                {
                    "name": name,
                    "kind": schema["properties"][name].get("x-kind"),
                    "allowed_values": schema["properties"][name].get("enum"),
                    "unit": schema["properties"][name].get("x-unit"),
                    "training_range": schema["properties"][name].get("x-observed-range"),
                    **_derived_info(uc, name),
                }
                for name in schema["x-feature-order"]
            ],
            "joins": _describe_joins(uc),
            "policy": uc.manifest.policy.model_dump(),
            "scenario": _describe_scenario(uc, schema),
            "feature_schema": schema,
        }

    def evaluation(self, use_case_id: str) -> dict[str, Any]:
        """How the model was chosen and how it does on the test split (for the dashboard)."""
        uc = self._ready(use_case_id)
        descriptor = uc.model.descriptor
        return {
            "use_case_id": use_case_id,
            "model_version": uc.model.version,
            "task_type": descriptor["task_type"],
            "algorithm": descriptor["algorithm"],
            "target_name": descriptor.get("target_name"),
            "target_unit": descriptor.get("target_unit"),
            "metrics": descriptor.get("metrics"),
            "candidates": descriptor.get("candidates"),
            "reference": descriptor.get("reference"),
            "feature_importance": descriptor.get("feature_importance"),
            "split": descriptor.get("split"),
            "sample": descriptor.get("evaluation"),
            "outputs": {name: _summary(om.descriptor, "metrics", "candidates", "reference", "feature_importance", "split")
                        | {"sample": om.descriptor.get("evaluation")} for name, om in uc.model.outputs.items()},
        }

    # --------------------------------------------------------------- scenarios

    def check_scenarios(self, use_case_id: str, base: dict[str, Any], scenarios: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate a scenario request and return exactly what would be calculated."""
        uc = self._scenario_use_case(use_case_id)
        resolved = self._resolve(uc, base.get("features"), scenarios)
        return {
            **self._header(uc),
            "base": {"record_id": base.get("record_id"), "features": resolved.base, "warnings": resolved.base_warnings},
            "scenarios": [{"name": s.name, "changes": s.changes, "features": s.features, "warnings": s.warnings}
                          for s in resolved.scenarios],
        }

    def run_scenarios(self, use_case_id: str, base: dict[str, Any], scenarios: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate the base and every scenario with the model, in one batch."""
        uc = self._scenario_use_case(use_case_id)
        started = time.perf_counter()
        resolved = self._resolve(uc, base.get("features"), scenarios)
        rows = [resolved.base] + [s.features for s in resolved.scenarios]
        outputs = self._outputs(uc, rows)
        sure = self._certainty(uc, rows)
        self._add_ranges(uc, outputs, sure, rows, resolved.assumed)
        self._monitor(uc, started, predicted=len(rows), rejected=0,
                      warnings=sum(len(s.warnings) for s in resolved.scenarios), rows=rows)
        return {
            **self._header(uc),
            "base": {"record_id": base.get("record_id"), "features": resolved.base,
                     "warnings": resolved.base_warnings, "outputs": outputs[0], "certainty": sure[0]},
            **({"assumed": _assumed_note(uc, resolved.assumed)} if resolved.assumed else {}),
            "scenarios": [self._scenario_result(uc, s.name, s.changes, s.features, s.warnings,
                                                resolved.base, outputs[0], out, c)
                          for s, out, c in zip(resolved.scenarios, outputs[1:], sure[1:])],
        }

    @staticmethod
    def _scenario_result(uc: UseCase, name: str, changes: list, features: dict, warnings: list,
                         base_row: dict, base_out: dict, out: dict, sure: dict | None) -> dict[str, Any]:
        cfg, descriptor = uc.manifest.scenario, _scenario_descriptors(uc)
        comparison = compare(cfg, descriptor, base_row, base_out, features, out)
        return {"name": name, "changes": changes, "features": features, "warnings": warnings, "outputs": out,
                "difference": comparison["outputs"], "net_eur": comparison["net_eur"], "certainty": sure,
                "summary": summarise(cfg, descriptor, name, comparison, sure)}

    @staticmethod
    def _certainty(uc: UseCase, rows: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        """How many comparable training cases each row has (None when not available).

        Every output has its own map (its rows, its important inputs). With several outputs a row shows the weakest
        one, so the message never promises more than the least certain output, plus the numbers per output.
        """
        if not uc.supports:
            return [None] * len(rows)
        matrix = uc.validator.to_matrix(rows)
        label = uc.manifest.scenario.case_label
        per_output = {name: uc.supports[name].comparable(matrix) for name in uc.supports}
        unknown_site = _unknown_site_note(uc)
        result = []
        for i in range(len(rows)):
            counts = {name: counts[i] for name, counts in per_output.items()}
            weakest = certainty(min(counts.values()), label)
            if weakest["level"] == "low" and unknown_site:
                weakest["message"] += unknown_site
            if len(counts) > 1:
                weakest["by_output"] = {name: {k: v for k, v in certainty(c, label).items() if k != "message"}
                                        for name, c in counts.items()}
            result.append(weakest)
        return result

    def sweep(self, use_case_id: str, base: dict[str, Any], lever: str, start: float | None = None,
              stop: float | None = None, step: float | None = None) -> dict[str, Any]:
        """Slide one lever over a range and calculate every point: a line instead of one number."""
        uc = self._scenario_use_case(use_case_id)
        started = time.perf_counter()
        values, errors = sweep_values(uc.manifest.scenario, lever, start, stop, step)
        if errors:
            raise RequestRejected(errors, detail="invalid scenario request")
        requested = [{"name": f"{lever} = {v}", "changes": {lever: {"set": v}}} for v in values]
        resolved = self._resolve(uc, base.get("features"), requested)
        rows = [resolved.base] + [s.features for s in resolved.scenarios]
        outputs = self._outputs(uc, rows)
        sure = self._certainty(uc, rows)
        self._add_ranges(uc, outputs, sure, rows, resolved.assumed)
        self._monitor(uc, started, predicted=len(rows), rejected=0, warnings=0, rows=rows)
        lever_info = next(l for l in describe_scenario(uc.manifest.scenario, uc.model.feature_schema,
                                                       _scenario_descriptors(uc))["levers"] if l["name"] == lever)
        seen = lever_info["training_range"]
        return {
            **self._header(uc),
            "lever": lever_info,
            "base": {"record_id": base.get("record_id"), "features": resolved.base, "warnings": resolved.base_warnings,
                     "value": resolved.base[lever], "outputs": outputs[0], "certainty": sure[0]},
            "points": [{"value": v, "outputs": out,
                        "difference": compare(uc.manifest.scenario, _scenario_descriptors(uc), resolved.base, outputs[0],
                                              s.features, out)["outputs"],
                        "beyond_training_data": bool(seen) and not seen[0] <= v <= seen[1],
                        "certainty": c}
                       for v, s, out, c in zip(values, resolved.scenarios, outputs[1:], sure[1:])],
        }

    def best_plan(self, use_case_id: str, base: dict[str, Any], levers: list[str] | None = None,
                  objective: dict[str, Any] | None = None, min_certainty: str = "low", top: int = 5,
                  min_gain: float | None = None) -> dict[str, Any]:
        """Try every setting of the levers and propose the plan with the best result, if it is worth mentioning."""
        uc = self._scenario_use_case(use_case_id)
        started = time.perf_counter()
        scenario, descriptors = uc.manifest.scenario, _scenario_descriptors(uc)
        levers = levers or list(scenario.levers)
        problems = [{"scenario": None, "field": name, "message": f"{name!r} cannot be changed in a scenario; you can change: "
                                                                 f"{', '.join(scenario.levers)}"}
                    for name in levers if name not in scenario.levers]
        outputs_known = {om.descriptor.get("target_name") or om.name for om in uc.model.outputs.values()}
        priced = [name for name, config in scenario.outputs.items() if config.value_eur is not None or config.price_from]
        if objective is None and not priced:
            problems.append({"scenario": None, "field": "objective", "message": "no output of this use case has a price, so there "
                             "is no net result to maximise; give an objective, e.g. {\"output\": \"...\", \"goal\": \"highest\"}"})
        if objective is not None and objective["output"] not in outputs_known:
            problems.append({"scenario": None, "field": "objective", "message": f"{objective['output']!r} is not an output; "
                             f"choose from: {', '.join(sorted(outputs_known))}"})
        if problems:
            raise RequestRejected(problems, detail="invalid scenario request")

        plans = planning.grid(scenario, levers)
        requested = [{"name": f"plan {i + 1}", "changes": {k: {"set": v} for k, v in plan.items()}} for i, plan in enumerate(plans)]
        resolved = self._resolve(uc, base.get("features"), requested)
        rows = [resolved.base] + [s.features for s in resolved.scenarios]
        outputs = self._outputs(uc, rows)
        sure = self._certainty(uc, rows)
        self._add_ranges(uc, outputs, sure, rows, resolved.assumed)
        self._monitor(uc, started, predicted=len(rows), rejected=0, warnings=0, rows=rows)
        properties = uc.model.feature_schema["properties"]

        def score(comparison: dict[str, Any]) -> float:
            if objective is None:
                return float(comparison["net_eur"] or 0.0)
            change = comparison["outputs"][objective["output"]].get("change", 0.0)
            return float(change if objective["goal"] == "highest" else -change)

        candidates, low_certainty, outside = [], 0, 0
        for plan, s, out, c in zip(plans, resolved.scenarios, outputs[1:], sure[1:]):
            comparison = compare(scenario, descriptors, resolved.base, outputs[0], s.features, out)
            beyond = ranges.outside_training(properties, s.features)
            level = c["level"] if c else None
            ok = planning.allowed(level, min_certainty) and not beyond
            low_certainty += not planning.allowed(level, min_certainty)
            outside += bool(beyond)
            candidates.append({"plan": plan, "scenario": s, "out": out, "certainty": c, "score": score(comparison),
                               "comparison": comparison, "ok": ok})
        candidates.sort(key=lambda item: -item["score"])
        trusted = [item for item in candidates if item["ok"]]

        def short(item: dict[str, Any]) -> dict[str, Any]:
            return {"plan": item["plan"], "gain": round(item["score"], 4), "net_eur": item["comparison"]["net_eur"],
                    "certainty": item["certainty"]["level"] if item["certainty"] else None}

        def full(item: dict[str, Any], name: str) -> dict[str, Any]:
            s = item["scenario"]
            return {**self._scenario_result(uc, name, s.changes, s.features, s.warnings, resolved.base, outputs[0],
                                            item["out"], item["certainty"]),
                    "plan": item["plan"], "gain": round(item["score"], 4)}

        # A gain must be worth mentioning: at least a share of the model's typical error in the objective's unit.
        margins = {name: margin_of_error(d) for name, d in _descriptors_by_output(uc).items()}
        prices = {name: _price(config, resolved.base) for name, config in scenario.outputs.items()}
        typical = planning.typical_error_of_objective(margins, prices, objective)
        threshold = min_gain if min_gain is not None else (planning.MIN_GAIN_SHARE * typical if typical else 0.0)
        top_trusted = trusted[0] if trusted and trusted[0]["score"] > 0 else None
        best = full(top_trusted, "best plan") if top_trusted and top_trusted["score"] >= threshold else None
        unlimited = candidates[0] if candidates and candidates[0]["score"] > 0 and not candidates[0]["ok"] else None
        if best:
            advice = f"Best plan for {planning.describe_objective(objective)}: " + best["summary"].split(": ", 1)[1]
        elif top_trusted:
            advice = (f"Keep the current settings. The best plan found would gain {top_trusted['score']:.2g}, less than the "
                      f"{threshold:.2g} the model can tell apart from its own error.")
        elif trusted:
            advice = f"Keep the current settings: no plan beats them for {planning.describe_objective(objective)}."
        else:
            advice = ("The model has too few comparable cases for every plan it tried, so it cannot recommend one. "
                      "Lower min_certainty to see what it would guess.")
        return {
            **self._header(uc),
            "objective": objective or {"output": "net_eur", "goal": "highest"},
            "levers": levers,
            "searched": {"plans": len(plans), "trusted": len(trusted), "left_out_low_certainty": int(low_certainty),
                         "left_out_outside_training_data": int(outside), "min_certainty": min_certainty,
                         "min_gain": round(threshold, 4), "typical_error": round(typical, 4) if typical else None},
            "base": {"record_id": base.get("record_id"), "features": resolved.base, "warnings": resolved.base_warnings,
                     "outputs": outputs[0], "certainty": sure[0]},
            "best": best,
            "alternatives": [short(item) for item in trusted[1: 1 + top] if item["score"] >= threshold] if best else [],
            # Found, but its gain is too small to tell apart from the model's error.
            "best_below_min_gain": short(top_trusted) if top_trusted and not best else None,
            # Shown, not recommended: what the model would pick if it ignored how little it knows there.
            "best_without_limits": ({**short(unlimited), "warning": "not recommended: the model has too few comparable cases "
                                     "for this plan or it lies outside its training data"} if unlimited else None),
            "advice": advice,
        }

    def _scenario_use_case(self, use_case_id: str) -> UseCase:
        uc = self._ready(use_case_id)
        if uc.manifest.scenario is None:
            raise RequestRejected(
                [{"scenario": None, "field": "scenario",
                  "message": f"use case '{use_case_id}' has no scenario levers; add a 'scenario' section to its manifest"}],
                detail="invalid scenario request",
            )
        return uc

    def _resolve(self, uc: UseCase, features: Any, scenarios: list[dict[str, Any]]) -> ResolvedRequest:
        def validate(row: Any) -> tuple[list[FieldError], list[str]]:
            errors = check_record(uc.manifest.policy, row)
            return (errors, []) if errors else uc.validator.validate(row)

        # The base is enriched once (joined columns from the lookup tables); every scenario starts from that
        # row, so a lever change is never overwritten by a join. The policy sees the raw request first.
        raw_errors = check_record(uc.manifest.policy, features)
        features, join_errors, join_warnings = _enrich(uc, features)
        problems = raw_errors or join_errors
        if problems:
            raise RequestRejected([{"scenario": None, "field": e.field, "message": e.message} for e in problems],
                                  detail="invalid scenario request")
        features, assumed = _assume(uc, features)
        resolved, errors = resolve_scenarios(uc.manifest.scenario, uc.model.feature_schema, features, scenarios, validate)
        if errors:
            raise RequestRejected(errors, detail="invalid scenario request")
        resolved.base_warnings[:0] = join_warnings
        resolved.assumed = assumed
        return resolved

    def _outputs(self, uc: UseCase, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Model outputs per row, keyed by output name: one input matrix, every output's own model."""
        matrix = uc.validator.to_matrix(rows)
        results: list[dict[str, Any]] = [{} for _ in rows]
        for om in uc.model.outputs.values():
            output = om.runner.predict(matrix)
            target = om.descriptor.get("target_name") or om.name
            for i in range(len(rows)):
                if om.descriptor["task_type"] == "classification":
                    results[i][target] = {"value": output.values[i],
                                          "probabilities": {k: round(v, 6) for k, v in output.probabilities[i].items()}}
                else:
                    value, warning = self._bounded(om.descriptor, float(output.values[i]))
                    results[i][target] = {"value": round(value, 6), **({"warning": warning} if warning else {})}
        return results

    @staticmethod
    def _add_ranges(uc: UseCase, outputs: list[dict[str, Any]], sure: list[dict[str, Any] | None],
                    rows: list[dict[str, Any]], assumed: list[str] | None = None) -> None:
        """Put the 80% range next to every regression value, as wide as the row's certainty asks for.

        A row outside the values the model was trained on gets no range: its error there was never measured.
        Where the caller left an unknown input open (`assumed`, e.g. the sun ahead), every value also gets an
        outlook (what it would be in a dark and in a sunny year) and the range is made wide enough to hold both.
        """
        properties = uc.model.feature_schema["properties"]
        outside = [ranges.outside_training(properties, row) for row in rows]
        outlook = _outlook(uc, rows, assumed)
        for om in uc.model.outputs.values():
            spec = om.descriptor.get("range")
            target = om.descriptor.get("target_name") or om.name
            for i, (out, c, beyond) in enumerate(zip(outputs, sure, outside)):
                alternatives = outlook.get(om.name, [{}] * len(rows))[i]
                if alternatives:
                    out[target]["outlook"] = alternatives
                if not spec:
                    continue
                if beyond:
                    out[target]["range"], out[target]["range_note"] = None, ranges.unavailable_note(beyond)
                    continue
                level = (c.get("by_output", {}).get(om.name) or c)["level"] if c else None
                around = ranges.around(spec, out[target]["value"], level, om.descriptor.get("target_bounds"),
                                       also=list(alternatives.values()))
                if around:
                    out[target]["range"] = around

    @staticmethod
    def _bounded(descriptor: dict[str, Any], value: float) -> tuple[float, str | None]:
        """Keep a regression output within the target's physical limits (e.g. gas >= 0)."""
        bounds = descriptor.get("target_bounds")
        if not bounds:
            return value, None
        low, high = bounds
        if low is not None and value < low:
            return low, f"the model predicted {value:g}, below the possible minimum {low:g}; shown as {low:g}"
        if high is not None and value > high:
            return high, f"the model predicted {value:g}, above the possible maximum {high:g}; shown as {high:g}"
        return value, None

    # -------------------------------------------------------------- prediction

    def predict(self, use_case_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        """All-or-nothing prediction for a JSON request."""
        uc = self._ready(use_case_id)
        started = time.perf_counter()
        checked = [self._check(uc, record, index) for index, record in enumerate(records)]
        invalid = [c for c in checked if c.errors]
        if invalid:
            self._monitor(uc, started, predicted=0, rejected=len(invalid), warnings=0)
            raise RequestRejected(
                [
                    {"record_id": c.record_id, "field": e.field, "message": e.message}
                    for c in invalid
                    for e in c.errors
                ]
            )
        predictions = self._run(uc, checked)
        warnings = sum(len(c.warnings) for c in checked)
        self._monitor(uc, started, predicted=len(checked), rejected=0, warnings=warnings, checked=checked)
        return {**self._header(uc), "predictions": predictions}

    def predict_stream(
        self, use_case_id: str, records: Iterable[Any], start_index: int = 0
    ) -> Iterator[dict[str, Any]]:
        """Record-by-record prediction for JSONL batches, in bounded chunks.

        Invalid records do not stop the batch; they produce an error line.
        """
        uc = self._ready(use_case_id)
        chunk: list[_Checked] = []
        for index, record in enumerate(records, start=start_index):
            chunk.append(self._check(uc, record, index))
            if len(chunk) >= STREAM_CHUNK_SIZE:
                yield from self._predict_chunk(uc, chunk)
                chunk = []
        if chunk:
            yield from self._predict_chunk(uc, chunk)

    def _predict_chunk(self, uc: UseCase, chunk: list[_Checked]) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        valid = [c for c in chunk if not c.errors]
        results = iter(self._run(uc, valid)) if valid else iter(())
        warnings = sum(len(c.warnings) for c in valid)
        self._monitor(
            uc, started, predicted=len(valid), rejected=len(chunk) - len(valid), warnings=warnings, checked=valid
        )
        for c in chunk:
            if c.errors:
                yield {
                    "record_id": c.record_id,
                    "errors": [{"field": e.field, "message": e.message} for e in c.errors],
                }
            else:
                yield {**next(results), "model_version": uc.model.version}

    # ---------------------------------------------------------------- helpers

    def _ready(self, use_case_id: str) -> UseCase:
        uc = self._use_cases.get(use_case_id)
        if uc is None:
            raise UseCaseNotFound(use_case_id)
        if uc.model is None:
            raise UseCaseNotReady(use_case_id)
        return uc

    def _check(self, uc: UseCase, record: Any, index: int) -> _Checked:
        if isinstance(record, UnparsableRecord):
            return _Checked(str(index), None, [FieldError("record", record.message)], [])
        if not isinstance(record, dict) or "features" not in record:
            return _Checked(str(index), None, [FieldError("record", "expected an object with 'features'")], [])
        record_id = str(record.get("record_id") if record.get("record_id") is not None else index)
        features = record["features"]
        errors = check_record(uc.manifest.policy, features)
        if errors:
            # Policy violations are reported before anything else touches the data.
            return _Checked(record_id, features, errors, [])
        sent = features
        features, join_errors, join_warnings = _enrich(uc, features)
        if join_errors:
            return _Checked(record_id, features, join_errors, [])
        features, assumed = _assume(uc, features)
        filled = _filled_in(uc, sent, features)
        errors = check_record(uc.manifest.policy, features)  # e.g. a group size that was joined in
        if errors:
            return _Checked(record_id, features, errors, [])
        errors, warnings = uc.validator.validate(features)
        return _Checked(record_id, features, errors, [*join_warnings, *warnings], filled, assumed)

    def _run(self, uc: UseCase, checked: list[_Checked]) -> list[dict[str, Any]]:
        """One prediction per record. One output keeps the flat shape; several give an ``outputs`` map."""
        matrix = uc.validator.to_matrix([c.features for c in checked])
        several = len(uc.model.outputs) > 1
        results: list[dict[str, Any]] = [{"record_id": c.record_id} for c in checked]
        # Certainty level per output and row (where the model has a map of its training data): sets the range's width.
        levels = {name: [certainty(n)["level"] for n in index.comparable(matrix)] for name, index in (uc.supports or {}).items()}
        properties = uc.model.feature_schema["properties"]
        outside = [ranges.outside_training(properties, c.features) for c in checked]
        # Unknown inputs the callers left open: per output and record, the value in e.g. a dark and a sunny year.
        open_inputs = sorted({name for c in checked for name in (c.assumed or [])})
        outlook = _outlook(uc, [c.features for c in checked], open_inputs)
        for om in uc.model.outputs.values():
            output = om.runner.predict(matrix)
            task = om.descriptor["task_type"]
            for i, c in enumerate(checked):
                part: dict[str, Any] = {}
                if task == "regression":
                    value, warning = self._bounded(om.descriptor, float(output.values[i]))
                    part["prediction"] = round(value, 6)
                    alternatives = ({k: v for k, v in outlook[om.name][i].items()}
                                    if om.name in outlook and c.assumed else {})
                    if alternatives:
                        part["outlook"] = alternatives
                    if om.descriptor.get("range") and outside[i]:
                        part["range"], part["range_note"] = None, ranges.unavailable_note(outside[i])
                    else:
                        around = ranges.around(om.descriptor.get("range"), value, levels[om.name][i] if om.name in levels else None,
                                               om.descriptor.get("target_bounds"), also=list(alternatives.values()))
                        if around:
                            part["range"] = around
                    if warning:
                        c.warnings.append(f"{om.name}: {warning}" if several else warning)
                elif task == "classification":
                    part["prediction"] = output.values[i]
                    part["probabilities"] = {k: round(v, 6) for k, v in output.probabilities[i].items()}
                else:
                    part["prediction"] = output.values[i]
                    part["anomaly_score"] = round(float(output.scores[i]), 6)
                if several:
                    results[i].setdefault("outputs", {})[om.name] = part
                else:
                    results[i].update(part)
        for item, c in zip(results, checked):
            item["warnings"] = c.warnings
            if c.assumed:
                item["assumed"] = _assumed_note(uc, c.assumed)
            if c.filled:
                item["filled_in"] = c.filled
        return results

    def _header(self, uc: UseCase) -> dict[str, Any]:
        descriptor = uc.model.descriptor
        return {
            "schema_version": SCHEMA_VERSION,
            "use_case_id": uc.manifest.use_case_id,
            "model_version": uc.model.version,
            "task_type": descriptor["task_type"],
            "target_name": descriptor.get("target_name"),
            "target_unit": descriptor.get("target_unit"),
            "outputs": {name: _summary(om.descriptor) for name, om in uc.model.outputs.items()},
        }

    def _monitor(
        self, uc: UseCase, started: float, *, predicted: int, rejected: int, warnings: int,
        checked: list[_Checked] | None = None, rows: list[dict[str, Any]] | None = None,
    ) -> None:
        features = None
        if uc.manifest.policy.log_feature_values and (checked or rows):
            features = rows if rows is not None else [c.features for c in checked]
        self.monitor.record(
            uc.manifest.use_case_id,
            predicted=predicted,
            rejected=rejected,
            warnings=warnings,
            latency_ms=(time.perf_counter() - started) * 1000,
            model_version=uc.model.version if uc.model else None,
            features=features,
        )


# ------------------------------------------------------------------------------ joined inputs

def _enrich(uc: UseCase, features: Any) -> tuple[Any, list[FieldError], list[str]]:
    """Fill the columns the model was trained with from lookup tables (no-op without joins or for non-objects)."""
    joins = uc.model.joins if uc.model else None
    if joins is None or not isinstance(features, dict):
        return features, [], []
    return joins.enrich_record(features, set(uc.validator.feature_order))


def _check_joined_levers(manifest: Manifest, model: LoadedModel) -> None:
    if model.joins is None or manifest.scenario is None:
        return
    for lever in manifest.scenario.levers:
        if lever in model.joins.derived:
            raise ManifestError(f"{manifest.use_case_id}: scenario lever {lever!r} is calculated by join "
                                f"{model.joins.derived[lever]!r}, so it cannot be changed; change its inputs instead")


def _derived_info(uc: UseCase, name: str) -> dict[str, Any]:
    joins = uc.model.joins
    if joins is None or name not in joins.derived:
        return {}
    return {"derived_by": joins.derived[name], "required": False}


def _filled_in(uc: UseCase, sent: Any, enriched: Any) -> dict[str, Any] | None:
    """The columns the joins added to a request (None when nothing was filled in)."""
    joins = uc.model.joins if uc.model else None
    if joins is None or not isinstance(sent, dict):
        return None
    filled = {c: enriched[c] for c in joins.derived if c in enriched and sent.get(c) is None}
    return filled or None


def _describe_joins(uc: UseCase) -> list[dict[str, Any]]:
    """For clients: how the model's inputs are connected to lookup tables and what can be sent instead."""
    return describe_joins(uc.model.joins, uc.model.descriptor, set(uc.model.feature_schema["properties"]),
                          uc.manifest.training.units)


# ------------------------------------------------------------------------------ several outputs

def _summary(descriptor: dict[str, Any], *extra: str) -> dict[str, Any]:
    """The few descriptor keys a client needs per output (plus the ``extra`` ones)."""
    keys = ("task_type", "algorithm", "target_name", "target_unit", *extra)
    return {key: descriptor.get(key) for key in keys}


def _scenario_descriptors(uc: UseCase) -> dict[str, Any]:
    """What the scenario code needs: the descriptor (one output) or target name -> descriptor (several)."""
    return uc.descriptors if len(uc.model.outputs) > 1 else uc.model.descriptor


def _check_outputs_match(manifest: Manifest, model: LoadedModel) -> None:
    """A use case that declares several outputs must be served from a version that has exactly those."""
    declared = set(manifest.training.outputs)
    if declared and declared != set(model.outputs):
        raise ManifestError(f"{manifest.use_case_id}: the trained version has outputs {sorted(model.outputs)} but the "
                            f"manifest declares {sorted(declared)}; retrain with `predictor train {manifest.use_case_id}`")
    for alias, output in manifest.aliases.items():
        if output not in model.outputs:
            raise ManifestError(f"{manifest.use_case_id}: alias {alias!r} points at output {output!r}, which the trained "
                                f"version does not have; retrain")


def _supports(manifest: Manifest, model: LoadedModel | None) -> dict[str, SupportIndex] | None:
    """The comparable-cases map of every output (keyed by target name), when scenarios need it and it was stored."""
    if model is None or manifest.scenario is None:
        return None
    found = {om.descriptor.get("target_name") or om.name: SupportIndex(om.support)
             for om in model.outputs.values() if om.support}
    return found or None


def _unknown_site_note(uc: UseCase) -> str | None:
    """For low certainty: what to expect if the model has no history of this site (measured on sites left out in training)."""
    if len(uc.model.outputs) != 1:
        return None
    descriptor = next(iter(uc.model.outputs.values())).descriptor
    width, group = (descriptor.get("range") or {}).get("new_group_half_width"), (descriptor.get("new_group") or {}).get("column")
    if width is None or not group:
        return None
    unit = f" {descriptor['target_unit']}" if descriptor.get("target_unit") else ""
    return (f" If the model has no history of this {group}, expect it to be off by up to about {width:.2g}{unit}; "
            f"log a season for this {group} first to rely on it.")


def _descriptors_by_output(uc: UseCase) -> dict[str, dict[str, Any]]:
    return {om.descriptor.get("target_name") or om.name: om.descriptor for om in uc.model.outputs.values()}


def _price(config: Any, row: dict[str, Any]) -> float | None:
    """Euros per unit of an output for this request (None when the output has no price)."""
    if config.value_eur is not None:
        return float(config.value_eur)
    if config.price_from and row.get(config.price_from) is not None:
        return float(row[config.price_from])
    return None


# ------------------------------------------------------------------------------ unknown inputs (weather ahead)

def _assume(uc: UseCase, features: Any) -> tuple[Any, list[str]]:
    """Set the unknown inputs the caller left out to their usual value; return which ones."""
    scenario = uc.manifest.scenario
    if scenario is None or not scenario.assumptions or not isinstance(features, dict):
        return features, []
    left_out = [name for name in scenario.assumptions if features.get(name) is None]
    if not left_out:
        return features, []
    return {**features, **{name: scenario.assumptions[name].typical for name in left_out}}, left_out


def _outlook(uc: UseCase, rows: list[dict[str, Any]], assumed: list[str] | None) -> dict[str, list[dict[str, float]]]:
    """Per regression output and row: the value at the low and the high end of every assumption left open."""
    if not assumed or not rows:
        return {}
    assumptions = uc.manifest.scenario.assumptions
    variants = [(a.low_label, name, a.low) for name, a in assumptions.items() if name in assumed]
    variants += [(a.high_label, name, a.high) for name, a in assumptions.items() if name in assumed]
    result: dict[str, list[dict[str, float]]] = {}
    for label, name, value in variants:
        matrix = uc.validator.to_matrix([{**row, name: value} for row in rows])
        for om in uc.model.outputs.values():
            if om.descriptor["task_type"] != "regression":
                continue
            values = om.runner.predict(matrix).values
            per_row = result.setdefault(om.name, [{} for _ in rows])
            for i in range(len(rows)):
                per_row[i][label] = round(PredictionService._bounded(om.descriptor, float(values[i]))[0], 6)
    return result


def _assumed_note(uc: UseCase, assumed: list[str]) -> dict[str, Any]:
    assumptions = uc.manifest.scenario.assumptions
    return {name: {"label": assumptions[name].label, "value": assumptions[name].typical,
                   "note": f"not given, so the usual value is used; 'outlook' shows {assumptions[name].low_label} "
                           f"({assumptions[name].low:g}) and {assumptions[name].high_label} ({assumptions[name].high:g})"}
            for name in assumed}


def _describe_scenario(uc: UseCase, schema: dict[str, Any]) -> dict[str, Any] | None:
    described = describe_scenario(uc.manifest.scenario, schema, _scenario_descriptors(uc))
    if described is not None and uc.manifest.scenario.assumptions:
        # Inputs a caller may leave out: the service then uses the usual value and shows the outlook.
        described["assumptions"] = [{"name": name, **a.model_dump()} for name, a in uc.manifest.scenario.assumptions.items()]
    return described
