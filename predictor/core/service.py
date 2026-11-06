"""The prediction service: identical for every use case.

Flow per request: policy check -> feature-schema validation -> conversion to a
typed matrix -> model runner -> normalised JSON result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from predictor.core.manifest import Manifest
from predictor.core.monitoring import Monitor
from predictor.core.policy import check_record, check_schema
from predictor.core.scenario import (
    ResolvedRequest,
    check_scenario,
    compare,
    describe_scenario,
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

    @property
    def status(self) -> str:
        return "ready" if self.model else "not_trained"


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
                validator = FeatureValidator(
                    model.feature_schema, reject_unknown_fields=manifest.policy.reject_unknown_fields
                )
            use_cases.append(UseCase(manifest, model, validator))
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
            "inputs": [
                {
                    "name": name,
                    "kind": schema["properties"][name].get("x-kind"),
                    "allowed_values": schema["properties"][name].get("enum"),
                    "unit": schema["properties"][name].get("x-unit"),
                    "training_range": schema["properties"][name].get("x-observed-range"),
                }
                for name in schema["x-feature-order"]
            ],
            "policy": uc.manifest.policy.model_dump(),
            "scenario": describe_scenario(uc.manifest.scenario, schema, descriptor),
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
        self._monitor(uc, started, predicted=len(rows), rejected=0,
                      warnings=sum(len(s.warnings) for s in resolved.scenarios), rows=rows)
        return {
            **self._header(uc),
            "base": {"record_id": base.get("record_id"), "features": resolved.base,
                     "warnings": resolved.base_warnings, "outputs": outputs[0]},
            "scenarios": [self._scenario_result(uc, s.name, s.changes, s.features, s.warnings,
                                                resolved.base, outputs[0], out)
                          for s, out in zip(resolved.scenarios, outputs[1:])],
        }

    @staticmethod
    def _scenario_result(uc: UseCase, name: str, changes: list, features: dict, warnings: list,
                         base_row: dict, base_out: dict, out: dict) -> dict[str, Any]:
        cfg, descriptor = uc.manifest.scenario, uc.model.descriptor
        comparison = compare(cfg, descriptor, base_row, base_out, features, out)
        return {"name": name, "changes": changes, "features": features, "warnings": warnings, "outputs": out,
                "difference": comparison["outputs"], "net_eur": comparison["net_eur"],
                "summary": summarise(cfg, descriptor, name, comparison)}

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
        self._monitor(uc, started, predicted=len(rows), rejected=0, warnings=0, rows=rows)
        lever_info = next(l for l in describe_scenario(uc.manifest.scenario, uc.model.feature_schema,
                                                       uc.model.descriptor)["levers"] if l["name"] == lever)
        seen = lever_info["training_range"]
        return {
            **self._header(uc),
            "lever": lever_info,
            "base": {"record_id": base.get("record_id"), "features": resolved.base, "warnings": resolved.base_warnings,
                     "value": resolved.base[lever], "outputs": outputs[0]},
            "points": [{"value": v, "outputs": out,
                        "difference": compare(uc.manifest.scenario, uc.model.descriptor, resolved.base, outputs[0],
                                              s.features, out)["outputs"],
                        "beyond_training_data": bool(seen) and not seen[0] <= v <= seen[1]}
                       for v, s, out in zip(values, resolved.scenarios, outputs[1:])],
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

        resolved, errors = resolve_scenarios(uc.manifest.scenario, uc.model.feature_schema, features, scenarios, validate)
        if errors:
            raise RequestRejected(errors, detail="invalid scenario request")
        return resolved

    def _outputs(self, uc: UseCase, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Model outputs per row, keyed by output name (one output now; more with multi-output)."""
        output = uc.model.runner.predict(uc.validator.to_matrix(rows))
        target = uc.model.descriptor.get("target_name")
        if uc.model.descriptor["task_type"] == "classification":
            return [{target: {"value": output.values[i],
                              "probabilities": {k: round(v, 6) for k, v in output.probabilities[i].items()}}}
                    for i in range(len(rows))]
        results = []
        for i in range(len(rows)):
            value, warning = self._bounded(uc, float(output.values[i]))
            results.append({target: {"value": round(value, 6), **({"warning": warning} if warning else {})}})
        return results

    @staticmethod
    def _bounded(uc: UseCase, value: float) -> tuple[float, str | None]:
        """Keep a regression output within the target's physical limits (e.g. gas >= 0)."""
        bounds = uc.model.descriptor.get("target_bounds")
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
        errors, warnings = uc.validator.validate(features)
        return _Checked(record_id, features, errors, warnings)

    def _run(self, uc: UseCase, checked: list[_Checked]) -> list[dict[str, Any]]:
        matrix = uc.validator.to_matrix([c.features for c in checked])
        output = uc.model.runner.predict(matrix)
        task = uc.model.descriptor["task_type"]
        results = []
        for i, c in enumerate(checked):
            item: dict[str, Any] = {"record_id": c.record_id}
            if task == "regression":
                value, warning = self._bounded(uc, float(output.values[i]))
                item["prediction"] = round(value, 6)
                if warning:
                    c.warnings.append(warning)
            elif task == "classification":
                item["prediction"] = output.values[i]
                item["probabilities"] = {k: round(v, 6) for k, v in output.probabilities[i].items()}
            else:
                item["prediction"] = output.values[i]
                item["anomaly_score"] = round(float(output.scores[i]), 6)
            item["warnings"] = c.warnings
            results.append(item)
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
