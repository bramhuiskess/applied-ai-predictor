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
from predictor.core.ports import LoadedModel, ModelRegistry
from predictor.core.validation import FeatureValidator, FieldError

SCHEMA_VERSION = "1.0"
STREAM_CHUNK_SIZE = 500


class UseCaseNotFound(Exception):
    pass


class UseCaseNotReady(Exception):
    pass


class RequestRejected(Exception):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__(f"{len(errors)} invalid record(s)")
        self.errors = errors


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
            "feature_schema": schema,
        }

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
                item["prediction"] = round(float(output.values[i]), 6)
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
        checked: list[_Checked] | None = None,
    ) -> None:
        features = None
        if uc.manifest.policy.log_feature_values and checked:
            features = [c.features for c in checked]
        self.monitor.record(
            uc.manifest.use_case_id,
            predicted=predicted,
            rejected=rejected,
            warnings=warnings,
            latency_ms=(time.perf_counter() - started) * 1000,
            model_version=uc.model.version if uc.model else None,
            features=features,
        )
