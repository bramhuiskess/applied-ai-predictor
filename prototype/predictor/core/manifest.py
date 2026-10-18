"""Use-case manifests: the only place where use-case knowledge lives.

A manifest is configuration, so it is validated strictly (unknown keys are
rejected) to limit configuration debt (Sculley et al., 2015).
"""

from __future__ import annotations

import json
import re
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


class TrainingConfig(_Strict):
    target: str | None = None
    task: Literal["auto", "regression", "classification", "anomaly"] = "auto"
    time_column: str | None = None
    horizon_steps: int | None = Field(default=None, ge=1)
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
        return self


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

    @model_validator(mode="after")
    def _special_implies_personal(self) -> Policy:
        if self.special_category_data and not self.contains_personal_data:
            raise ValueError("special_category_data requires contains_personal_data")
        if self.contains_personal_data and self.log_feature_values:
            raise ValueError("feature values of personal data may not be logged")
        return self


class Manifest(_Strict):
    manifest_version: str
    use_case_id: str
    description: str = ""
    data_source: DataSource
    training: TrainingConfig
    model: ModelRef = ModelRef()
    policy: Policy = Policy()

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
