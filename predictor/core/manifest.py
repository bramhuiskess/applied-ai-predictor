"""Use-case manifests: the only place where use-case knowledge lives.

A manifest is configuration, so it is validated strictly (unknown keys are
rejected) to limit configuration debt (Sculley et al., 2015).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MANIFEST_FILENAME = "manifest.json"
SUPPORTED_MANIFEST_VERSIONS = {"1.0"}


class ManifestError(Exception):
    """Raised when a manifest is invalid or references something unknown."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSource(_Strict):
    adapter: str
    path: str | None = None


class Baseline(_Strict):
    """A column that is itself a simple guess for the target, to compare the model against."""

    column: str
    label: str


class TestGroup(_Strict):
    """Hold out whole groups (a farm, a greenhouse) as the test set: can the model handle one it never saw?"""

    column: str
    values: list[str] = Field(min_length=1)


class TrainingConfig(_Strict):
    target: str | None = None
    task: Literal["auto", "regression", "classification", "anomaly"] = "auto"
    time_column: str | None = None
    horizon_steps: int | None = Field(default=None, ge=1)
    # Backtest: everything from this date on is the test set (e.g. train up to 2020, test 2021-2025).
    test_from: date | None = None
    baseline: Baseline | None = None
    test_group: TestGroup | None = None
    # Physical limits of the target (gas use cannot be negative). Predictions are kept within them.
    target_min: float | None = None
    target_max: float | None = None
    drop_columns: list[str] = []
    rename_columns: dict[str, str] = {}
    evaluation_label: str | None = None
    anomaly_share: float = Field(default=0.05, gt=0.0, le=0.5)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=0.9)
    units: dict[str, str] = {}

    @model_validator(mode="after")
    def _check_consistency(self) -> TrainingConfig:
        if self.task == "anomaly" and self.target:
            raise ValueError("an anomaly use case has no target; use 'evaluation_label' to score it")
        if self.task in ("regression", "classification") and not self.target:
            raise ValueError(f"task '{self.task}' needs a 'target'")
        if self.horizon_steps and not self.time_column:
            raise ValueError("'horizon_steps' needs a 'time_column' to order the records")
        if self.baseline and (self.horizon_steps or self.task in ("classification", "anomaly")):
            raise ValueError("'baseline' is for regression without horizon_steps (that has its own baseline)")
        if (self.target_min is not None or self.target_max is not None) and self.task not in ("auto", "regression"):
            raise ValueError("'target_min'/'target_max' only apply to regression")
        if self.target_min is not None and self.target_max is not None and self.target_min >= self.target_max:
            raise ValueError("'target_min' must be smaller than 'target_max'")
        if self.test_group and self.test_from:
            raise ValueError("use either 'test_from' or 'test_group' to choose the test set, not both")
        if self.test_from and not self.time_column:
            raise ValueError("'test_from' needs a 'time_column' to know which records come after it")
        return self

    @property
    def outputs(self) -> list[str]:
        """What the use case predicts. One target for now; the multi-output ticket extends this."""
        return [self.target] if self.target else []


class ModelRef(_Strict):
    runner: str = "onnx"
    version: str = "latest"


class MinGroupSize(_Strict):
    field: str
    min: int = Field(ge=1)


class Policy(_Strict):
    contains_personal_data: bool = False
    special_category_data: bool = False
    reject_unknown_fields: bool = True
    log_feature_values: bool = False
    forbidden_fields: list[str] = []
    min_group_size: MinGroupSize | None = None
    # Row-level test results (actual vs. predicted) for the dashboard. Default: only
    # without personal data; set true explicitly for e.g. synthetic or public data.
    publish_evaluation_rows: bool | None = None

    @property
    def evaluation_rows_allowed(self) -> bool:
        if self.publish_evaluation_rows is None:
            return not self.contains_personal_data
        return self.publish_evaluation_rows

    @model_validator(mode="after")
    def _special_implies_personal(self) -> Policy:
        if self.special_category_data and not self.contains_personal_data:
            raise ValueError("special_category_data requires contains_personal_data")
        if self.contains_personal_data and self.log_feature_values:
            raise ValueError("feature values of personal data may not be logged")
        return self


MAX_LEVER_STEPS = 10_000


class Lever(_Strict):
    """An input the user may change in a scenario (e.g. the greenhouse temperature)."""

    label: str | None = None
    unit: str | None = None
    min: float
    max: float
    step: float = Field(gt=0)
    # Expected direction per output, only where it always holds ("more lamp light -> less gas").
    # It becomes a hard monotone constraint for the model, so do not declare what you cannot back up.
    effects: dict[str, Literal["increasing", "decreasing"]] = {}

    @model_validator(mode="after")
    def _check_range(self) -> Lever:
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be smaller than max ({self.max})")
        if (self.max - self.min) / self.step > MAX_LEVER_STEPS:
            raise ValueError(f"step {self.step} is too small for the range {self.min}-{self.max}")
        return self


class ScenarioOutput(_Strict):
    label: str | None = None
    # Value of one unit of this output in euros, for comparing scenarios in money.
    # Positive for what you sell (a plant), negative for what you pay (a m3 of gas).
    value_eur: float | None = None
    # Or take the price per unit from an input of the same row (the grower's own gas price).
    price_from: str | None = None
    is_cost: bool = False  # with price_from: the output costs money instead of earning it

    @model_validator(mode="after")
    def _one_price(self) -> ScenarioOutput:
        if self.value_eur is not None and self.price_from:
            raise ValueError("give either 'value_eur' or 'price_from', not both")
        if self.is_cost and not self.price_from:
            raise ValueError("'is_cost' goes with 'price_from' (with value_eur, use a negative value for a cost)")
        return self


class ScenarioConfig(_Strict):
    levers: dict[str, Lever] = Field(min_length=1)
    outputs: dict[str, ScenarioOutput] = {}
    # What one row is, in the user's words ("batches", "fields"), for messages like
    # "based on 14 comparable batches in the history".
    case_label: str = Field(default="cases", min_length=1, max_length=40)


class Manifest(_Strict):
    manifest_version: str
    use_case_id: str
    description: str = ""
    data_source: DataSource
    training: TrainingConfig
    model: ModelRef = ModelRef()
    policy: Policy = Policy()
    scenario: ScenarioConfig | None = None

    @model_validator(mode="after")
    def _check_scenario(self) -> Manifest:
        if self.scenario is None:
            return self
        t = self.training
        if not t.outputs or t.task == "anomaly":
            raise ValueError("scenario: needs a target to predict (anomaly use cases have no output to compare)")
        if t.horizon_steps:
            raise ValueError("scenario: not supported together with horizon_steps yet")
        not_features = {
            "target": set(t.outputs), "time_column": {t.time_column}, "drop_columns": set(t.drop_columns),
            "evaluation_label": {t.evaluation_label},
        }
        forbidden = {f.lower() for f in self.policy.forbidden_fields}
        for name, lever in self.scenario.levers.items():
            for what, names in not_features.items():
                if name in names:
                    raise ValueError(f"scenario.levers.{name}: is the {what}, so it is not a model input")
            if name.lower() in forbidden:
                raise ValueError(f"scenario.levers.{name}: is a forbidden field in the policy")
            for output in lever.effects:
                if output not in t.outputs:
                    raise ValueError(f"scenario.levers.{name}.effects: unknown output {output!r} "
                                     f"(outputs: {', '.join(t.outputs)})")
        for output, config in self.scenario.outputs.items():
            if output not in t.outputs:
                raise ValueError(f"scenario.outputs: unknown output {output!r} (outputs: {', '.join(t.outputs)})")
            if config.price_from and config.price_from in t.outputs:
                raise ValueError(f"scenario.outputs.{output}.price_from: {config.price_from!r} is an output, not an input")
        return self

    @field_validator("manifest_version")
    @classmethod
    def _supported_version(cls, value: str) -> str:
        if value not in SUPPORTED_MANIFEST_VERSIONS:
            raise ValueError(f"unsupported manifest_version {value!r}")
        return value

    @field_validator("use_case_id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", value):
            raise ValueError("use_case_id must be a lowercase slug, e.g. 'greenhouse-climate'")
        return value


def load_manifest(path: Path) -> Manifest:
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"{path}: file not found") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON ({exc})") from exc
    try:
        manifest = Manifest.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in exc.errors()
        )
        raise ManifestError(f"{path}: {problems}") from exc
    if path.parent.name != manifest.use_case_id:
        raise ManifestError(
            f"{path}: use_case_id {manifest.use_case_id!r} must match its directory name {path.parent.name!r}"
        )
    return manifest


def load_all_manifests(usecases_dir: Path) -> list[Manifest]:
    usecases_dir = Path(usecases_dir)
    manifests = [load_manifest(p) for p in sorted(usecases_dir.glob(f"*/{MANIFEST_FILENAME}"))]
    return manifests


def check_references(manifest: Manifest, *, known_sources: set[str], known_runners: set[str]) -> None:
    """Fail fast when a manifest names an adapter that is not installed."""
    if manifest.data_source.adapter not in known_sources:
        raise ManifestError(
            f"{manifest.use_case_id}: unknown data_source adapter {manifest.data_source.adapter!r} "
            f"(available: {', '.join(sorted(known_sources))})"
        )
    if manifest.model.runner not in known_runners:
        raise ManifestError(
            f"{manifest.use_case_id}: unknown model runner {manifest.model.runner!r} "
            f"(available: {', '.join(sorted(known_runners))})"
        )
